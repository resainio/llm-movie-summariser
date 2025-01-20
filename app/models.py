from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    """Schema for the input"""
    review: str = Field(..., title="Review", description="The text of the movie review.")
    reviewer: str = Field(..., title="Reviewer", description="The name of the movie reviewer.")

class MovieSummary(BaseModel):
    """Schema for the output"""
    title: str = Field(..., title="Title", description="A concise title for the movie review.")
    summary: str = Field(..., title="Summary", description="A short summary of the movie review.")
    grade: int = Field(..., ge=0, le=5, title="Grade", description="A grade for the movie review.")
    reviewer: str = Field(..., title="Reviewer", description="The name of the movie reviewer.")
