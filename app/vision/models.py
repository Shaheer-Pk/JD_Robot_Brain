from sqlalchemy import Column, Integer, JSON, DateTime, func

# ADJUST THIS IMPORT: point it to wherever your shared SQLAlchemy Base actually
# lives in app/users/database.py
from app.users.database import Base


class FaceEmbedding(Base):
    
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)   # links back to users.id as foreign key
    embedding = Column(JSON)                # the 128 numbers, stored as a list (embedding)
    created_at = Column(DateTime, server_default=func.now())