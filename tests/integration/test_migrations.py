"""Integration test for the Alembic migration chain.

The revisions contain MySQL-specific operations, so this test is SKIPPED on SQLite
(the local default) and runs in CI against the MySQL service. It proves the whole
chain applies and reverses cleanly — which also guards against the two ORM-less
tables (sentimental_resources, song_resources) drifting out of sync."""
import os

import pytest

pytestmark = pytest.mark.integration

_ON_SQLITE = os.environ.get("DATABASE_URL", "sqlite://").startswith("sqlite")


@pytest.mark.skipif(
    _ON_SQLITE,
    reason="Alembic revisions target MySQL; validated in CI against the MySQL service (TEST_DATABASE_URL).",
)
def test_migration_chain_upgrades_and_downgrades():
    from alembic import command
    from alembic.config import Config

    from db.session import Base, engine

    # Start from a clean schema (conftest created tables via create_all).
    Base.metadata.drop_all(bind=engine)

    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    # Restore the ORM-created schema so any later tests see the expected tables.
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
