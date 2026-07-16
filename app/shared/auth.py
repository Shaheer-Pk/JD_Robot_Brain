import bcrypt

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 20  # well under bcrypt's 72-byte hard limit — truncation is structurally impossible at this cap


def validate_password_length(password: str) -> None:
    if not (MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH):
        raise ValueError(
            f"Password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters."
        )


def hash_password(password: str) -> str:
    validate_password_length(password)
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")  # stored as TEXT in the users table


def verify_password(password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))