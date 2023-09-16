from typing import Optional
from pydantic import BaseModel


class CapModel(BaseModel):
    user_id: str
    name: Optional[str]
    description: Optional[str]
