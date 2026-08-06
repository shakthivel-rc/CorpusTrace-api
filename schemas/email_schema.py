from pydantic import BaseModel

class EmailTemplate(BaseModel):
    subject: str
    body: str