"""Pydantic schema for a structured restaurant record (M1)."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)
    itemId: Optional[int] = None
