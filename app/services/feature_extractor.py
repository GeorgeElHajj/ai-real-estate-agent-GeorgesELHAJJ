import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from google import genai
from pydantic import ValidationError
from datetime import datetime

from app.schemas.extraction import (
    ExtractionResult,
    empty_extraction,
)
from app.utils.prompt_logger import log_prompt_result

load_dotenv()


PROMPT_DIR = Path("prompts")


AGE_DESCRIPTOR_MAP = {
    "brand new": 0,
    "new": 2,
    "newer": 5,
    "modern": 8,
    "older": 20,
    "old": 30,
    "historic": 50,
}


def load_prompt(version: str) -> str:
    prompt_path = PROMPT_DIR / f"stage1_{version}.txt"
    return prompt_path.read_text(encoding="utf-8")


def normalize_house_age(result: ExtractionResult, reference_year: int | None = None) -> ExtractionResult:
    if reference_year is None:
        reference_year = datetime.now().year

    if result.features.HouseAge is None:
        if result.year_built_raw is not None:
            derived_age = reference_year - result.year_built_raw

            if derived_age >= 0:
                result.features.HouseAge = derived_age
                result.assumptions.append(
                    f"HouseAge derived from build year {result.year_built_raw} using reference year {reference_year}."
                )

        elif result.age_descriptor:
            descriptor = result.age_descriptor.strip().lower()

            for key, age_value in AGE_DESCRIPTOR_MAP.items():
                if key in descriptor:
                    result.features.HouseAge = age_value
                    result.assumptions.append(
                        f"HouseAge approximated from age descriptor '{descriptor}'."
                    )
                    break

    extracted_fields = []
    missing_fields = []

    for field_name, value in result.features.model_dump().items():
        if value is None:
            missing_fields.append(field_name)
        else:
            extracted_fields.append(field_name)

    result.extracted_fields = extracted_fields
    result.missing_fields = missing_fields
    result.needs_user_input = len(missing_fields) > 0

    return result

def call_gemini(prompt: str, model_name: str = os.getenv("GEMINI_MODEL")) -> str:
    client = genai.Client()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": ExtractionResult.model_json_schema(),
        },
    )

    return response.text


def extract_features(user_query: str, version: str = "v1") -> dict:
    prompt_template = load_prompt(version)
    prompt = prompt_template.replace("{{USER_QUERY}}", user_query)

    try:
        raw_output = call_gemini(prompt)

        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output.replace("```json", "", 1).strip()
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()

        parsed = ExtractionResult.model_validate(json.loads(cleaned_output))
        parsed = normalize_house_age(parsed)

        log_prompt_result({
            "prompt_version": version,
            "input_query": user_query,
            "raw_output": raw_output,
            "validated": True,
            "parsed_output": parsed.model_dump(),
        })

        return parsed.model_dump()

    except (ValidationError, json.JSONDecodeError, ValueError, Exception) as e:
        fallback = empty_extraction(prompt_version=version, error=str(e))

        log_prompt_result({
            "prompt_version": version,
            "input_query": user_query,
            "raw_output": None,
            "validated": False,
            "error": str(e),
            "parsed_output": fallback.model_dump(),
        })

        return fallback.model_dump()