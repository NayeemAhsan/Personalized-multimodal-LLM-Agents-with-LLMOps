from pydantic import BaseModel, Field
from typing import List, Optional

class RealEstateListing(BaseModel):
    neighborhood: Optional[str] = None
    price: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    square_footage: Optional[str] = None
    description: Optional[str] = None
    neighborhood_description: Optional[str] = None
    image_path: Optional[str] = None

class ListingCollection(BaseModel):
    listings: List[RealEstateListing] = Field(description="List of available real estate")
