from pydantic import BaseModel


class CapModel(BaseModel):
    name: str | None
    description: str | None
