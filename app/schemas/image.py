from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict

from app.schemas.extraction import ExtractedFeatures


class ImageGenerationRequest(BaseModel):
    query: str
    features: ExtractedFeatures


class ImageGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"]
    message: str
    image_prompt: str
    image_base64: Optional[str] = None