from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict

from app.schemas.extraction import ExtractedFeatures
from app.schemas.response import PredictionSummary


class QueryRequest(BaseModel):
    query: str


class IncompleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["incomplete"]
    message: str
    extracted_features: ExtractedFeatures
    extracted_fields: List[str]
    missing_fields: List[str]
    assumptions: List[str]
    prompt_version: str


class CompleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete"]
    message: str
    extracted_features: ExtractedFeatures
    extracted_fields: List[str]
    missing_fields: List[str]
    assumptions: List[str]
    prediction: PredictionSummary
    interpretation: Optional[str]
    model_name: Optional[str]
    extraction_prompt_version: str
    interpretation_prompt_version: str
    
class PredictFromFeaturesRequest(BaseModel):
    query: str
    features: ExtractedFeatures