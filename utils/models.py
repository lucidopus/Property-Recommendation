from typing import Optional
from pydantic import BaseModel, Field


class LeadRequest(BaseModel):
    leadid: Optional[str | int] = Field(
        default=None, description="Unique identifier of the lead project"
    )
    projectname: Optional[str] = Field(default=None, description="Name of the project")
    city: str = Field(
        description="Name of the city where the project is located",
    )
    location: Optional[str] = Field(
        default=None, description="Name of the micromarket area for the project"
    )
    sublocation: Optional[str] = Field(
        default=None, description="Name of the locality within the micromarket"
    )
    status: Optional[str] = Field(
        default=None, description="Current status of the project"
    )
    unit_type: str = Field(
        description="Specifies whether the project is Residential or Commercial"
    )
    unit_cat_name: str = Field(
        description="Category of unit, such as Apartment, Studio, Plot, etc"
    )
    furnishing_status: Optional[str] = Field(
        default=None, description="Specifies the furnishing status of the Apartment"
    )
    latitude: float = Field(description="Latitude coordinate of the project's location")
    longitude: float = Field(
        description="Longitude coordinate of the project's location"
    )
    unit_size: float = Field(
        description="Area of the project or unit, typically in square feet or meters"
    )
    bedroom_count: float = Field(
        description="Number of bedrooms in the unit (for residential projects)"
    )
    unit_price: Optional[float] = Field(
        default=None, description="Budget or price range of the project or unit"
    )
    unit_rent: Optional[float] = Field(
        default=None, description="Rent of the project or unit"
    )


class SuccessResponse(BaseModel):
    success: bool = Field(
        default=True,
        description="Determines if the operation was carried out successfully",
    )
    message: str = Field(
        default="MATCH FOUND",
        description="Further explains the status of the operation",
    )
    vector_distance: Optional[float] = Field(
        default=None,
        description="Distance between the lead and focus vectors obtained from Custom Manhattan Algorithm",
    )
    similarity_score: float = Field(
        description="Percentage of similarity between the lead project and the focus project"
    )
    projectid: str | int = Field(description="Unique identifier of the focus project")
    projectname: str = Field(
        description="Name of the project",
    )
    developer_name: Optional[str] = Field(
        default=None,
        description="Name of the project developer",
    )
    city: str = Field(
        description="Name of the city where the project is located",
    )
    location: str = Field(
        description="Name of the micromarket area for the project",
    )
    sublocation: str = Field(
        description="Name of the locality within the micromarket",
    )
    status: Optional[str] = Field(
        default=None,
        description="Current status of the project",
    )
    unit_type: str = Field(
        description="Specifies whether the project is Residential or Commercial"
    )
    unit_cat_name: str = Field(
        description="Category of unit, such as Apartment, Studio, Plot, etc"
    )
    furnishing_status: Optional[str] = Field(
        default=None, description="Specifies the furnishing status of the Apartment"
    )
    latitude: float = Field(description="Latitude coordinate of the project's location")
    longitude: float = Field(
        description="Longitude coordinate of the project's location"
    )
    distance: float = Field(
        description="Point distance between the lead project and the focus project"
    )
    unit_size: float = Field(
        description="Area of the project or unit, typically in square feet or meters"
    )
    bedroom_count: float = Field(
        description="Number of bedrooms in the unit (for residential projects"
    )
    unit_price: Optional[float] = Field(
        default=None, description="Budget or price range of the project or unit"
    )
    unit_rent: Optional[float] = Field(
        default=None, description="Rent of the project or unit"
    )


class FailResponse(BaseModel):
    success: bool = Field(
        default=False,
        description="Determines whether the operation was carried out successfully or not",
    )
    message: str = Field(
        default="NO MATCH FOUND",
        description="Further explains the status of the operation",
    )
