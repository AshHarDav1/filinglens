import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from filinglens.config import Settings


@pytest.fixture
def db_session():
    """A session on the dev Postgres, inside a transaction that is always rolled back.

    Skipped when the database is not running (`docker compose up -d db` + `alembic upgrade head`).
    """
    engine = create_engine(Settings(_env_file=None).database_url)
    try:
        connection = engine.connect()
        connection.execute(text("SELECT 1 FROM chunk LIMIT 1"))
    except OperationalError:
        pytest.skip("Postgres not available")
    connection.rollback()  # end the probe's implicit transaction
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()
