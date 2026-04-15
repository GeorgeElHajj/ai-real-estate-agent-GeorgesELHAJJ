from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


FEATURE_NAMES = [
    "OverallQual",
    "GrLivArea",
    "Neighborhood",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "LotArea",
    "BedroomAbvGr",
    "HouseStyle",
    "HouseAge",
]


class ExtractedFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OverallQual: Optional[int] = Field(default=None, ge=1, le=10)
    GrLivArea: Optional[float] = Field(default=None, ge=0)
    Neighborhood: Optional[str] = None
    TotalBsmtSF: Optional[float] = Field(default=None, ge=0)
    GarageCars: Optional[float] = Field(default=None, ge=0)
    FullBath: Optional[float] = Field(default=None, ge=0)
    LotArea: Optional[float] = Field(default=None, ge=0)
    BedroomAbvGr: Optional[int] = Field(default=None, ge=0)
    HouseStyle: Optional[str] = None
    HouseAge: Optional[int] = Field(default=None, ge=0)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: ExtractedFeatures
    extracted_fields: List[str]
    missing_fields: List[str]
    needs_user_input: bool
    assumptions: List[str] = []
    prompt_version: str

    # raw helper fields for post-processing
    year_built_raw: Optional[int] = Field(default=None, ge=1800, le=2100)
    age_descriptor: Optional[str] = None


class ExtractionFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: ExtractedFeatures
    extracted_fields: List[str]
    missing_fields: List[str]
    needs_user_input: bool
    assumptions: List[str]
    prompt_version: str
    error: str


def empty_extraction(prompt_version: str, error: str = "Extraction failed") -> ExtractionFailure:
    return ExtractionFailure(
        features=ExtractedFeatures(),
        extracted_fields=[],
        missing_fields=FEATURE_NAMES.copy(),
        needs_user_input=True,
        assumptions=["User review required before prediction."],
        prompt_version=prompt_version,
        error=error,
    )