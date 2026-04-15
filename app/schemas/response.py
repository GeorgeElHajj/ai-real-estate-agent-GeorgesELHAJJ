from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.extraction import ExtractedFeatures


class PredictionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_price: Optional[float] = Field(default=None, ge=0)
    formatted_price: Optional[str] = None
    median_price: Optional[float] = Field(default=None, ge=0)
    mean_price: Optional[float] = Field(default=None, ge=0)
    q1_price: Optional[float] = Field(default=None, ge=0)
    q3_price: Optional[float] = Field(default=None, ge=0)
    relative_position: Optional[str] = None


class Stage2Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: ExtractedFeatures
    missing_fields: List[str]
    needs_user_input: bool

    prediction: PredictionSummary
    interpretation: Optional[str] = None

    assumptions: List[str] = []
    interpretation_prompt_version: str
    model_name: Optional[str] = None