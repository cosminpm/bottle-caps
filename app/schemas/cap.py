from pydantic import BaseModel


class CapModel(BaseModel):
    user_id: str
    name: str | None
    description: str | None
