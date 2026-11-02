"""SQLAlchemy engine and session factory."""

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


def create_db_engine(settings: Settings) -> Engine:
    """Create a PostgreSQL engine from validated application settings."""
    return create_engine(
        settings.database_url.strip(),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        connect_args={"connect_timeout": settings.db_connect_timeout},
    )


def create_session_factory(engine: Engine) -> sessionmaker:
    """Create a non-expiring SQLAlchemy session factory for an engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


settings = Settings()
engine = create_db_engine(settings)
SessionLocal = create_session_factory(engine)


def get_db() -> Iterator[Session]:
    """Yield a database session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
