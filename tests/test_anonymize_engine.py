"""The anonymization engine and its deterministic lane (no deck, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from captain_claw.anonymize import detectors as d
from captain_claw.anonymize import engine as E
from captain_claw.anonymize import engine as engine_module
from captain_claw.anonymize import formats

CASES = json.loads((Path(engine_module.__file__).parent / "cases.json").read_text(encoding="utf-8"))


def _run(text, state=None, **opts):
    state = state if state is not None else E.VaultState()
    out, result = E.anonymize_text(text, state, E.Options(**opts))
    return out, result, state


# ── the shared fixture ──────────────────────────────────────────────


@pytest.mark.parametrize("case", [c for c in CASES["cases"] if "deterministic" in c["lanes"]], ids=lambda c: c["id"])
def test_shared_fixture_deterministic_lane(case):
    out, result, state = _run(case["text"])
    for value in case["must_hide"]:
        assert value not in out, f"{case['id']}: {value!r} survived:\n{out}"
    for value in case["must_keep"]:
        assert value in out, f"{case['id']}: {value!r} was removed:\n{out}"
    # Everything replaced is restorable from the vault.
    back, n, unknown = E.restore(out, state)
    assert unknown == [] and n == sum(e.count for e in result.used)


# ── detectors ───────────────────────────────────────────────────────


class TestChecksums:
    def test_iban(self):
        assert d.iban_ok("HR1210010051863000160")
        assert d.iban_ok("DE89 3704 0044 0532 0130 00")
        assert not d.iban_ok("HR1210010051863000161")  # one digit off
        assert not d.iban_ok("HR12100100518630001")  # wrong length for HR

    def test_oib_and_jmbg(self):
        assert d.oib_ok("52601815906") and not d.oib_ok("52601815907")
        assert d.jmbg_ok("1503985336097") and not d.jmbg_ok("1503985336098")
        assert not d.jmbg_ok("3213985336097")  # day 32

    def test_unvalidated_numbers_are_left(self):
        out, _r, _s = _run("Ref 52601815907 and HR1210010051863000161.")
        assert "52601815907" in out and "HR1210010051863000161" in out


class TestShapes:
    def kinds(self, text):
        return {(s.kind, text[s.start:s.end]) for s in d.detect(text)}

    def test_dates_and_times_are_not_phones(self):
        found = self.kinds("Rok 01.02.2023 u 08:30, račun 00012345, iznos 1.250,00.")
        assert not any(k == "PHONE" for k, _v in found)

    def test_phone_shapes(self):
        found = self.kinds("Zovi 091 234 5678, +385 1 4567 890 ili tel: 01/4567-890.")
        phones = {v for k, v in found if k == "PHONE"}
        assert {"091 234 5678", "+385 1 4567 890", "01/4567-890"} <= phones

    def test_version_numbers_are_not_ips(self):
        found = self.kinds("Version 1.2.3.4 runs on 10.0.0.7")
        assert ("IP", "10.0.0.7") in found and ("IP", "1.2.3.4") not in found

    def test_profile_links_skip_site_pages(self):
        found = self.kinds("facebook.com/sharer and facebook.com/ivan.horvat")
        assert ("PROFILE_URL", "facebook.com/ivan.horvat") in found
        assert not any(v.endswith("sharer") for _k, v in found)

    def test_company_label_stops_at_the_name(self):
        found = self.kinds("Tvrtka: Acme d.o.o. potpisuje. Ivan Horvat dolazi.")
        assert ("ORG", "Acme d.o.o.") in found

    def test_stop_words_are_not_names(self):
        assert not any(s.kind == "PERSON" for s in d.detect("Dear Team,\nThanks,\nThe Board"))

    def test_hints(self):
        assert d.hint_for("E-mail") == "EMAIL" and d.hint_for("Ime i prezime") == "PERSON"
        assert d.hint_for("Datum rođenja") == "DOB" and d.hint_for("iznos") == ""
        assert d.hint_span("Widget", "PERSON?") is None  # a bare "name" column: one word isn't enough
        assert d.hint_span("Ana Kovač", "PERSON?").kind == "PERSON"


# ── entities and tokens ─────────────────────────────────────────────


class TestEntities:
    def test_same_value_same_token_across_texts(self):
        state = E.VaultState()
        a, _r, _s = _run("Email ivan@x.hr or IBAN HR1210010051863000160.", state)
        b, _r, _s = _run("Again: IVAN@X.HR, HR12 1001 0051 8630 0016 0", state)
        assert a == "Email ⟦EMAIL_1⟧ or IBAN ⟦IBAN_1⟧."
        assert b == "Again: ⟦EMAIL_1⟧, ⟦IBAN_1⟧"

    def test_vaults_never_share_meanings(self):
        a, _r, _s = _run("Mail ana@x.hr")
        b, _r, _s = _run("Mail ivo@y.hr")
        assert a.endswith("⟦EMAIL_1⟧") and b.endswith("⟦EMAIL_1⟧")

    def test_phone_formats_are_one_number(self):
        out, result, _s = _run("tel. +385 91 234 5678 ili 091 234 5678 ili 00385912345678")
        assert out.count("⟦PHONE_1⟧") == 3 and len(result.used) == 1

    def test_croatian_inflections_are_one_person(self):
        text = ("Poštovani gospodine Horvat,\nugovor za Ivana Horvata stiže. Ivanu Horvatu javljam.\n"
                "Lijep pozdrav,\nIvan Horvat")
        out, result, state = _run(text)
        assert "Horvat" not in out and "Ivan" not in out
        persons = [e for e in result.used if e.kind == "PERSON"]
        assert len(persons) == 1
        assert persons[0].original == "Ivan Horvat"  # restore gives the plain form
        # The vault finds him next time with no context at all.
        later, _r, _s = _run("Horvatom smo zadovoljni, I. Horvat potpisuje.", state)
        assert later == "⟦PERSON_1⟧ smo zadovoljni, ⟦PERSON_1⟧ potpisuje."

    def test_two_people_sharing_a_surname(self):
        text = "Lijep pozdrav,\nIvan Horvat\n\nName: Ana Horvat\nHorvat je stigao."
        out, result, state = _run(text)
        tokens = {e.original: e.token for e in result.used}
        assert tokens["Ivan Horvat"] != tokens["Ana Horvat"]
        assert "Horvat" not in out  # the lone surname is "a Horvat" of its own
        lone = [e for e in result.used if e.key.startswith("~")]
        assert len(lone) == 1 and lone[0].original == "Horvat"

    def test_keep_and_extra(self):
        out, _r, _s = _run("Captain Labs d.o.o. and Acme d.o.o. run Project Falcon.",
                           keep=("Captain Labs",), extra=(("Project Falcon", "PROJECT"),))
        assert out == "Captain Labs d.o.o. and ⟦ORG_1⟧ run ⟦PROJECT_1⟧."

    def test_kinds_limit(self):
        out, _r, _s = _run("Ivan@x.hr, IBAN HR1210010051863000160", kinds=frozenset({"IBAN"}))
        assert "Ivan@x.hr" in out and "⟦IBAN_1⟧" in out

    def test_existing_tokens_are_left_alone(self):
        state = E.VaultState()
        first, _r, _s = _run("Mail ivan@x.hr", state)
        again, result, _s = _run(first + " and ana@x.hr", state)
        assert again == "Mail ⟦EMAIL_1⟧ and ⟦EMAIL_2⟧"


class TestRestore:
    def test_round_trip(self):
        text = "Name: Klaus Müller\nIBAN: DE89370400440532013000\nmail k.mueller@beispiel.de"
        out, _r, state = _run(text)
        back, n, unknown = E.restore(out, state)
        assert back == text and n == 3 and unknown == []

    def test_unknown_and_rewritten_tokens(self):
        out, _r, state = _run("mail ana@x.hr")
        back, n, unknown = E.restore("[[EMAIL_1]] and ⟦PERSON_9⟧", state)
        assert back == "ana@x.hr and ⟦PERSON_9⟧" and n == 1 and unknown == ["⟦PERSON_9⟧"]


class TestVerify:
    def test_fails_closed_when_data_keeps_turning_up(self, monkeypatch):
        """A detector that finds one more thing every round: after the last
        round something is still there, so nothing is returned."""
        def one_at_a_time(text, kinds=None, **kw):
            i = text.find("a")
            return [d.Span(i, i + 1, "PERSON", 2, "test")] if i >= 0 else []

        monkeypatch.setattr(d, "detect", one_at_a_time)
        with pytest.raises(E.VerifyFailed) as exc:
            E.anonymize_text("a a a a a", E.VaultState())
        assert exc.value.kinds == {"PERSON": 1}
        assert "a a" not in str(exc.value)  # never the data itself

    def test_second_look_is_counted(self):
        _out, result, _s = _run("mail ana@x.hr")
        assert result.rounds == 2 and result.fixed == 0


def test_filename():
    state = E.VaultState()
    _run("Lijep pozdrav,\nIvan Horvat", state)
    assert E.anonymize_filename("Ivan_Horvat_CV", state) == "PERSON_1_CV"
    assert E.anonymize_filename("report-2024", state) == "report_2024"


# ── formats ─────────────────────────────────────────────────────────


class TestFormats:
    def test_csv_cells_and_column_hints(self):
        content = "ime,prezime,email,grad,iznos\nIvan,Horvat,ivan@x.hr,Zagreb,100\nAna,Kovač,ana.kovac@y.com,Split,250\n"
        prep = formats.prepare(content, "csv")
        result = E.run(prep.segments, E.VaultState())
        out = prep.rebuild(result.outputs)
        rows = [line.split(",") for line in out.strip().split("\n")]
        assert rows[0] == ["ime", "prezime", "email", "grad", "iznos"]
        assert rows[1][3:] == ["Zagreb", "100"] and rows[2][3:] == ["Split", "250"]
        assert all(re.fullmatch(r"⟦\w+⟧", c) for c in rows[1][:3] + rows[2][:3])
        assert out.endswith("\n")

    def test_json_keys_numbers_and_shape(self):
        content = json.dumps({"client": {"full_name": "Ana Kovač", "oib": 52601815906, "city": "Split"},
                              "notes": ["call +385 91 234 5678"], "count": 3}, indent=2)
        prep = formats.prepare(content, "json")
        result = E.run(prep.segments, E.VaultState())
        data = json.loads(prep.rebuild(result.outputs))
        assert data["client"]["full_name"] == "⟦PERSON_1⟧"
        assert data["client"]["oib"] == "⟦OIB_1⟧"
        assert data["client"]["city"] == "Split" and data["count"] == 3
        assert data["notes"] == ["call ⟦PHONE_1⟧"]

    def test_bad_json(self):
        with pytest.raises(formats.FormatError):
            formats.prepare("{nope", "json")


# ── how agents reach the tool ───────────────────────────────────────


@pytest.mark.parametrize("message", [
    "anonymize this contract before you summarise it",
    "please redact the PII in clients.csv",
    "remove personal data from the report",
    "anonimiziraj ovaj ugovor",
    "ukloni osobne podatke iz tablice",
    "put the real names back into ⟦PERSON_1⟧'s letter",
])
def test_eco_mode_surfaces_the_tool(message):
    from captain_claw.agent_orchestration_mixin import _eco_select_tools_by_intent

    assert "anonymize" in _eco_select_tools_by_intent(message)


def test_tool_is_registered_for_every_agent():
    import inspect

    from captain_claw import agent_context_mixin, deck_confine
    from captain_claw.flight_deck import tool_risk

    src = inspect.getsource(agent_context_mixin)
    assert "self.tools.register(AnonymizeTool())" in src
    assert deck_confine.PATH_RULES["anonymize"] == (("path", "read"), ("output_path", "write"))
    assert deck_confine.URL_PARAMS["anonymize"] == ("url",)
    assert tool_risk._TOOL_RISK["anonymize"] == "write"


# ── P4: coarser values (one way) and the name-recognition hook ──────


class TestGeneralize:
    def _run(self, text, kinds=E.GENERALIZE_KINDS):
        opts = E.Options(generalize=frozenset(kinds))
        state = E.VaultState()
        seg = [E.Segment(text)]
        return E.generalize(seg, E.run(seg, state, opts), state, opts.generalize).outputs[0]

    def test_dates_addresses_ages_amounts(self):
        out = self._run("Klijent, rođen 12. 3. 1985., star 45 godina, adresa: Ilica 12, 10000 Zagreb. "
                        "Plaća 4.237,50 EUR, bonus €1.250 i $87,500.00.")
        assert "rođen 1985" in out and "40–49 godina" in out and "adresa: Zagreb" in out
        assert "2.000–5.000 EUR" in out and "€1.000–2.000" in out and "$50,000–100,000" in out
        for exact in ("12. 3. 1985", "45", "Ilica 12", "4.237,50", "1.250", "87,500"):
            assert exact not in out, exact

    def test_only_what_was_asked_and_tokens_otherwise(self):
        out = self._run("Rođen 12. 3. 1985., plaća 4.237,50 EUR, adresa: Ilica 12", kinds=("DOB",))
        assert "Rođen 1985" in out and "4.237,50 EUR" in out and "⟦ADDRESS_1⟧" in out
        # An address with no city in it keeps its token.
        assert "⟦ADDRESS_" in self._run("adresa: Ilica 12")

    def test_amounts_and_ages_are_not_private_by_default(self):
        out, _r, _s = _run("Plaća 4.237,50 EUR, star 45 godina.")
        assert out == "Plaća 4.237,50 EUR, star 45 godina."

    @pytest.mark.parametrize("raw,value", [("1.250,00", 1250.0), ("4,237.10", 4237.1), ("12 500", 12500.0),
                                           ("250", 250.0), ("1.250", 1250.0), ("12.50", 12.5)])
    def test_amount_formats(self, raw, value):
        assert E._parse_amount(raw)[0] == value

    def test_offices_get_the_coarse_value_where_it_stands(self):
        text = "Plaća: 4.237,50 EUR"
        opts = E.Options(generalize=frozenset({"AMOUNT"}))
        state = E.VaultState()
        res = E.generalize([E.Segment(text)], E.run([E.Segment(text)], state, opts), state, opts.generalize)
        assert E.apply_edits(text, res.edits[0]) == res.outputs[0] == "Plaća: 2.000–5.000 EUR"


class TestNamePack:
    @pytest.fixture(autouse=True)
    def _off_after(self):
        yield
        d.set_ner_backend(None)

    def test_bare_names_found_and_followed(self):
        def model(text):
            out = []
            for name in ("Marko", "Petra", "Zagreb"):
                i = text.find(name)
                if i >= 0:
                    out.append((i, i + len(name), "LOC" if name == "Zagreb" else "PER", 0.97))
            j = text.find("Ivan")
            if j >= 0:
                out.append((j, j + 4, "PER", 0.4))  # not sure enough
            return out

        d.set_ner_backend(model)
        state = E.VaultState()
        out, _r, _s = _run("We met Marko and Petra in Zagreb; Ivan came too.", state)
        assert out == "We met ⟦PERSON_1⟧ and ⟦PERSON_2⟧ in Zagreb; Ivan came too."
        later, _r, _s = _run("Marku i Petri šaljem ugovor.", state)  # the vault knows them now
        assert "Marku" not in later and "Petri" not in later

    def test_the_model_reads_only_the_original_text(self):
        seen = []

        def model(text):
            seen.append(text)
            i = text.find("Marko")
            return [(i, i + 5, "PER", 0.97)] if i >= 0 else []

        d.set_ner_backend(model)
        out, _r, _s = _run("⟦PERSON_7⟧ met Marko; Marko paid.")
        assert out == "⟦PERSON_7⟧ met ⟦PERSON_1⟧; ⟦PERSON_1⟧ paid."
        # Once, on the first round: never on the engine's own masked output
        # (where it took the words around a token for names), and a token
        # already in the text reaches it as spaces, not NULs.
        assert seen == ["           met Marko; Marko paid."]

    def test_company_hits_follow_the_kinds_asked(self):
        def model(text):
            i = text.find("Northwind")
            return [(i, i + 9, "ORG", 0.97)] if i >= 0 else []

        d.set_ner_backend(model)
        assert [s.kind for s in d.detect("We signed with Northwind.")] == ["ORG"]
        assert d.detect("We signed with Northwind.", kinds={"PERSON"}) == []
        assert d.detect("We signed with Northwind.", ner=False) == []

    def test_a_broken_model_finds_nothing(self):
        def broken(text):
            raise RuntimeError("model crashed")

        d.set_ner_backend(broken)
        out, _r, _s = _run("We met Marko; mail ana@x.hr")
        assert out == "We met Marko; mail ⟦EMAIL_1⟧"


@pytest.mark.parametrize("text,address", [
    ("adresa: Ilica 12, 10000 Zagreb. Plaća je 4.000 EUR.", "Ilica 12, 10000 Zagreb"),
    ("Adresa: Trg b. J. Jelačića 1, 10000 Zagreb. Sastanak u 9.", "Trg b. J. Jelačića 1, 10000 Zagreb"),
    ("Address: 221 Baker Street, London; phone later", "221 Baker Street, London"),
])
def test_address_label_stops_at_the_sentence(text, address):
    found = {text[s.start:s.end] for s in d.detect(text) if s.kind == "ADDRESS" and s.source == "address-label"}
    assert found == {address}
