import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Points at database.db file sitting right next to this file
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False is required for SQLite + FastAPI 
# (different threads can use the same connection — needed since we run_in_threadpool elsewhere)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Shared Base — both app/users/models.py and app/vision/models.py import this
Base = declarative_base()


def get_db():
    # FastAPI dependency — gives each request its own DB session, then closes it.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()