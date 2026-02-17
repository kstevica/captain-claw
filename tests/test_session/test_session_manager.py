import pytest

from captain_claw.session import SessionManager


@pytest.mark.asyncio
async def test_session_manager_uses_db_path_override(tmp_path):
    db_path = tmp_path / "custom-sessions.db"
    manager = SessionManager(db_path=db_path)
    try:
        await manager.create_session(name="alpha")
        assert db_path.exists()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_save_load_and_select_session_round_trip(tmp_path):
    manager = SessionManager(db_path=tmp_path / "sessions.db")
    try:
        created = await manager.create_session(name="alpha", metadata={"project": "captain-claw"})
        created.add_message("user", "hello")
        created.add_message("assistant", "hi")
        await manager.save_session(created)

        loaded = await manager.load_session(created.id)
        assert loaded is not None
        assert loaded.id == created.id
        assert loaded.name == "alpha"
        assert loaded.metadata["project"] == "captain-claw"
        assert len(loaded.messages) == 2

        selected_by_id = await manager.select_session(created.id)
        assert selected_by_id is not None
        assert selected_by_id.id == created.id

        selected_by_name = await manager.select_session("alpha")
        assert selected_by_name is not None
        assert selected_by_name.id == created.id
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_list_sessions_orders_by_last_update(tmp_path):
    manager = SessionManager(db_path=tmp_path / "sessions.db")
    try:
        first = await manager.create_session(name="one")
        second = await manager.create_session(name="two")

        first.add_message("user", "most recent")
        await manager.save_session(first)

        sessions = await manager.list_sessions(limit=10)
        assert len(sessions) == 2
        assert sessions[0].id == first.id
        assert sessions[1].id == second.id
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_delete_session_returns_status(tmp_path):
    manager = SessionManager(db_path=tmp_path / "sessions.db")
    try:
        session = await manager.create_session(name="to-delete")
        assert await manager.delete_session(session.id) is True
        assert await manager.delete_session(session.id) is False
    finally:
        await manager.close()
