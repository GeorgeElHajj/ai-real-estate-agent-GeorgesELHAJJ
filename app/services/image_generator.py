import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from app.schemas.extraction import ExtractedFeatures
from app.schemas.image import ImageGenerationResponse


load_dotenv()

PROMPT_DIR = Path("prompts")


def load_image_prompt(version: str = "v1") -> str:
    prompt_path = PROMPT_DIR / f"image_prompt_{version}.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_house_image_prompt(
    query: str,
    features: ExtractedFeatures,
    version: str = "v1",
) -> str:
    template = load_image_prompt(version)
    features_json = json.dumps(features.model_dump(), indent=2)

    prompt = template.replace("{{USER_QUERY}}", query)
    prompt = prompt.replace("{{FEATURES_JSON}}", features_json)

    return prompt


def generate_visual_prompt_with_gemini(
    prompt: str,
    model_name: Optional[str] = None,
) -> str:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    model = model_name or os.getenv("GEMINI_MODEL")

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text.strip()


def call_image_provider(image_prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = client.images.generate(
        model=os.getenv("GPT_IMAGE_MODEL"),
        prompt=image_prompt,
        size="1024x1024",
    )

    return result.data[0].b64_json


def fallback_visual_prompt(features: ExtractedFeatures) -> str:
    parts = [
        "A realistic front exterior architectural visualization of a residential house"
    ]

    if features.HouseStyle:
        style_map = {
            "1Story": "single-story house",
            "2Story": "two-story house",
            "SLvl": "split-level house",
            "SFoyer": "split-foyer house",
        }
        parts.append(style_map.get(features.HouseStyle, f"{features.HouseStyle} style house"))

    if features.BedroomAbvGr is not None:
        parts.append(f"with approximately {int(features.BedroomAbvGr)} bedrooms")

    if features.FullBath is not None:
        parts.append(f"and {int(features.FullBath)} full bathrooms")

    if features.GarageCars is not None:
        parts.append(f"with an attached {int(features.GarageCars)}-car garage")

    if features.GrLivArea is not None:
        parts.append(f"around {int(features.GrLivArea)} square feet of living space")

    if features.TotalBsmtSF is not None and features.TotalBsmtSF > 0:
        parts.append("with a basement")

    if features.LotArea is not None:
        parts.append(f"on a lot of about {int(features.LotArea)} square feet")

    if features.HouseAge is not None:
        if features.HouseAge <= 5:
            parts.append("appearing newly built")
        elif features.HouseAge <= 20:
            parts.append("appearing modern and well maintained")
        else:
            parts.append("appearing older but well maintained")

    if features.OverallQual is not None:
        if features.OverallQual >= 8:
            parts.append("with premium high-end finishes")
        elif features.OverallQual >= 6:
            parts.append("with good quality finishes")
        else:
            parts.append("with simple modest finishes")

    if features.Neighborhood:
        parts.append(f"in a suburban neighborhood similar to {features.Neighborhood}")

    parts.append("daylight, realistic materials, front view, high quality, no text, no watermark")

    return ", ".join(parts)


def generate_house_image(
    query: str,
    features: ExtractedFeatures,
    version: str = "v1",
) -> ImageGenerationResponse:
    base_prompt = build_house_image_prompt(query, features, version=version)

    try:
        image_prompt = generate_visual_prompt_with_gemini(base_prompt)
    except Exception as e:
        image_prompt = fallback_visual_prompt(features)

    try:
        image_base64 = call_image_provider(image_prompt)
        return ImageGenerationResponse(
            status="success",
            message="Image generated successfully.",
            image_prompt=image_prompt,
            image_base64=image_base64,
        )
    except Exception as e:
        return ImageGenerationResponse(
            status="error",
            message=str(e),
            image_prompt=image_prompt,
            image_base64=None,
        )