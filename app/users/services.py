from sqlalchemy.orm import Session
 
from app.users.models import User
 
 
def get_user_profile(db: Session, user_id: int) -> dict | None:
    """
    Looks up a user by id and returns the fields the brain module needs
    to build a personalized prompt: name, persona, memory_context.
 
    Returns None if no user with that id exists (vision's identify_face
    should fall back to the Guest persona in that case).
    """
    user = db.query(User).filter(User.id == user_id).first()
 
    if user is None:
        return None
 
    return {
        "id": user.id,
        "name": user.name,
        "persona": user.persona,
        "memory_context": user.memory_context,
    }
 