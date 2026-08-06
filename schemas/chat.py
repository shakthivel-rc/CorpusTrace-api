from pydantic import BaseModel, Field


class ResourceRenameRequest(BaseModel):
    # Bounded to the column width: `resources.resource_name` is VARCHAR(255) and MySQL
    # raises error 1406 rather than truncating, which would surface as an unhandled 500.
    resource_name: str = Field(..., min_length=1, max_length=255)
