import re

def to_snake_case(name: str) -> str:
    """Converts a string to snake_case."""
    name = re.sub(r'[\s-]+', '_', name)  # Replace spaces and hyphens with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)  # Remove special characters
    return name.lower()