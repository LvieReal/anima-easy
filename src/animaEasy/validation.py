from animaEasy.errors import ValidationError


def requireText(value: object, label: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValidationError(f"{label} must be text of at most {limit} characters without null characters.")
    return value
