from sqlalchemy import Column, Integer, String
 
from app.users.database import Base
 
 
class User(Base):
    """
    Maps to the existing `users` table in database.db.
    Columns must match exactly what setup_db.py already created:
    id, name, persona, memory_context.
    """
    __tablename__ = "users"
 
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    persona = Column(String, nullable=True)
    memory_context = Column(String, nullable=True)
 