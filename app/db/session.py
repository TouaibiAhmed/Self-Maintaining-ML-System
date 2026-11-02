"""SQLAlchemy engine and session factory."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings

settings = Settings()
engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    connect_args={"connect_timeout": settings.db_connect_timeout},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """Yield a database session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
