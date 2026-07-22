"""Deep memory scoring and relevance gating.

These are pure-unit tests over the scoring helpers — no Typesense required.
They exist because deep memory shipped for months with hybrid search silently
disabled and a score that reached the LLM as ``score=1157451470635796601.00``,
and nothing in the suite noticed.
"""

from captain_claw.deep_memory import _COLLECTION_SCHEMA_TEMPLATE, DeepMemoryIndex


def _hit(*, text_match=None, tokens_matched=None, dropped=0, distance=None, fused=None):
    """Build a Typesense hit in one of its three observed shapes."""
    hit: dict = {}
    if text_match is not None:
        hit["text_match"] = text_match
    if tokens_matched is not None:
        hit["text_match_info"] = {
            "tokens_matched": tokens_matched,
            "num_tokens_dropped": dropped,
        }
    if distance is not None:
        hit["vector_distance"] = distance
    if fused is not None:
        hit["hybrid_search_info"] = {"rank_fusion_score": fused}
    return hit


class TestNormalizeScore:
    def test_raw_text_match_never_leaks_through(self):
        """The regression: text_match is an unbounded ~19-digit integer.

        The old fusion did ``max(text_match, 1/(1+distance))``, so this value
        won every comparison and was formatted into the prompt verbatim.
        """
        hit = _hit(text_match=1157451470635796601, tokens_matched=2, dropped=0)
        score = DeepMemoryIndex._normalize_score(hit)
        assert 0.0 <= score <= 1.0

    def test_keyword_coverage_is_the_matched_fraction(self):
        assert DeepMemoryIndex._normalize_score(
            _hit(tokens_matched=1, dropped=1)
        ) == 0.5
        assert DeepMemoryIndex._normalize_score(
            _hit(tokens_matched=3, dropped=0)
        ) == 1.0

    def test_cosine_distance_becomes_similarity(self):
        # Typesense reports cosine *distance*; 0.4 distance = 0.6 similarity.
        assert DeepMemoryIndex._normalize_score(
            _hit(distance=0.4)
        ) == pytest_approx(0.6)

    def test_orthogonal_and_opposed_vectors_floor_at_zero(self):
        assert DeepMemoryIndex._normalize_score(_hit(distance=1.0)) == 0.0
        assert DeepMemoryIndex._normalize_score(_hit(distance=1.8)) == 0.0

    def test_best_of_both_signals_wins(self):
        """A strong keyword match survives a weak vector, and vice versa."""
        # Weak vector (0.9 distance -> 0.1), full keyword coverage.
        assert DeepMemoryIndex._normalize_score(
            _hit(tokens_matched=2, dropped=0, distance=0.9)
        ) == 1.0
        # No keyword match at all, strong vector.
        assert DeepMemoryIndex._normalize_score(
            _hit(tokens_matched=0, dropped=2, distance=0.2)
        ) == pytest_approx(0.8)

    def test_positional_fusion_score_is_ignored(self):
        """rank_fusion_score decays as 0.3*(1/rank) — it measures position,
        not relevance, so an excellent third-place hit must not be scored 0.10.
        """
        hit = _hit(tokens_matched=0, dropped=2, distance=0.2, fused=0.10)
        assert DeepMemoryIndex._normalize_score(hit) == pytest_approx(0.8)

    def test_empty_hit_scores_zero(self):
        assert DeepMemoryIndex._normalize_score({}) == 0.0


class TestRelevanceFloor:
    def _index(self, floor=0.12):
        return DeepMemoryIndex(api_key="k", min_score=floor)

    def test_floor_is_inclusive(self):
        idx = self._index(floor=0.12)
        assert idx._passes_relevance(0.12, 0.12)
        assert not idx._passes_relevance(0.119, 0.12)

    def test_zero_floor_admits_everything_scored(self):
        idx = self._index()
        assert idx._passes_relevance(0.0, 0.0)

    def test_explicit_request_can_relax_the_floor(self):
        """The caller passes floor=0.0 when the user explicitly asked for the
        archive; a marginal hit then beats returning nothing.
        """
        idx = self._index(floor=0.5)
        assert not idx._passes_relevance(0.2, 0.5)
        assert idx._passes_relevance(0.2, 0.0)


class TestEmbeddingWidth:
    def test_configured_zero_means_probe_the_provider(self):
        """0 is the auto-detect sentinel — the config must not pin a width."""
        assert DeepMemoryIndex(api_key="k")._embedding_dims == 0

    def test_no_chain_reports_no_width(self):
        assert DeepMemoryIndex(api_key="k")._probe_chain_dims() == 0

    def test_probe_reads_the_actual_vector_length(self):
        class Chain:
            enabled = True

            def embed_batch(self, texts):
                return ("stub:256", [[0.0] * 256 for _ in texts])

        idx = DeepMemoryIndex(api_key="k", embedding_chain=Chain())
        assert idx._probe_chain_dims() == 256

    def test_disabled_vectors_short_circuit_embedding(self):
        class Chain:
            enabled = True

            def embed_batch(self, texts):  # pragma: no cover - must not run
                raise AssertionError("embed_batch called while vectors disabled")

        idx = DeepMemoryIndex(api_key="k", embedding_chain=Chain())
        idx._vectors_disabled = True
        assert idx._embed(["anything"]) == []

    def test_width_mismatch_latches_off_rather_than_retrying_forever(self):
        """A provider that changes width mid-run must disable vectors once,
        not discard a batch per call while logging at debug level.
        """
        class Chain:
            enabled = True

            def embed_batch(self, texts):
                return ("stub:128", [[0.0] * 128 for _ in texts])

        idx = DeepMemoryIndex(api_key="k", embedding_chain=Chain())
        idx._embedding_dims = 256  # collection says 256, provider gives 128
        assert idx._embed(["x"]) == []
        assert idx._vectors_disabled is True


def pytest_approx(value, tol=1e-9):
    """Local approx helper so the module has no pytest-version coupling."""

    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

        def __repr__(self):  # pragma: no cover - only on failure
            return f"~{value}"

    return _Approx()


class TestSchemaMigration:
    """An existing collection must grow new optional fields.

    Adding a field to the schema *template* only affects freshly created
    collections. An older collection keeps its old shape, and Typesense answers
    400 to any ``filter_by`` naming a field it doesn't have — so a missed
    migration breaks every owner-scoped read and delete, not just new writes.
    """

    def _index_with_live_fields(self, monkeypatch, live_names):
        idx = DeepMemoryIndex(api_key="k")
        sent = {}

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

        class Client:
            def patch(self, url, json=None, **_):
                sent["fields"] = json["fields"]
                return Resp()

        monkeypatch.setattr(idx, "_get_client", lambda: Client())
        added = idx._add_missing_fields({"fields": [{"name": n} for n in live_names]})
        return added, sent.get("fields", [])

    def test_missing_optional_fields_are_added(self, monkeypatch):
        added, sent = self._index_with_live_fields(
            monkeypatch, ["doc_id", "source", "reference", "path", "text",
                          "chunk_index", "updated_at", "embedding"]
        )
        assert "owner_id" in added
        assert "content_hash" in added
        assert all(f.get("optional") for f in sent)

    def test_required_fields_are_never_added(self, monkeypatch):
        """A required field has no value for documents already stored, so
        Typesense would reject the alter outright."""
        added, sent = self._index_with_live_fields(monkeypatch, ["owner_id", "content_hash"])
        for name in ("doc_id", "text", "chunk_index", "updated_at"):
            assert name not in added

    def test_up_to_date_collection_is_left_alone(self, monkeypatch):
        names = [f["name"] for f in _COLLECTION_SCHEMA_TEMPLATE["fields"]]
        added, sent = self._index_with_live_fields(monkeypatch, names)
        assert added == [] and sent == []


class TestUnownedCount:
    """Typesense has no null filter — the count comes from facet arithmetic."""

    def _count(self, monkeypatch, payload):
        idx = DeepMemoryIndex(api_key="k")
        idx._collection_ensured = True

        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [payload]}

        class Client:
            def post(self, *a, **k):
                return Resp()

        monkeypatch.setattr(idx, "_get_client", lambda: Client())
        return idx.unowned_count()

    def test_total_minus_owned(self, monkeypatch):
        got = self._count(monkeypatch, {
            "found": 11,
            "facet_counts": [{"field_name": "owner_id", "counts": [{"value": "alice", "count": 4}]}],
        })
        assert got == 7

    def test_no_facet_values_means_all_unowned(self, monkeypatch):
        assert self._count(monkeypatch, {"found": 11, "facet_counts": []}) == 11

    def test_fully_owned_is_zero(self, monkeypatch):
        got = self._count(monkeypatch, {
            "found": 6,
            "facet_counts": [{"field_name": "owner_id",
                              "counts": [{"value": "a", "count": 2}, {"value": "b", "count": 4}]}],
        })
        assert got == 0

    def test_never_returns_negative(self, monkeypatch):
        got = self._count(monkeypatch, {
            "found": 1,
            "facet_counts": [{"field_name": "owner_id", "counts": [{"value": "a", "count": 5}]}],
        })
        assert got == 0
