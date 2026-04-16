import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from google import genai

from app.schemas.extraction import ExtractedFeatures
from app.schemas.response import PredictionSummary, Stage2Response
from app.utils.prompt_logger import log_prompt_result


load_dotenv()

PROMPT_DIR = Path("prompts")


def load_prompt(version: str = "final") -> str:
    prompt_path = PROMPT_DIR / f"stage2_{version}.txt"
    return prompt_path.read_text(encoding="utf-8")


def call_gemini_text(prompt: str, model_name: str = os.getenv("GEMINI_MODEL"), retries: int = 3) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e


def build_interpretation_prompt(
    user_query: str,
    features: ExtractedFeatures,
    prediction: PredictionSummary,
    version: str = "final",
) -> str:
    prompt_template = load_prompt(version)

    features_json = json.dumps(features.model_dump(), indent=2)
    prediction_json = json.dumps(prediction.model_dump(), indent=2)

    prompt = prompt_template.replace("{{USER_QUERY}}", user_query)
    prompt = prompt.replace("{{FEATURES_JSON}}", features_json)
    prompt = prompt.replace("{{PREDICTION_JSON}}", prediction_json)

    return prompt


def fallback_interpretation(prediction: PredictionSummary) -> str:
    base = f"The estimated price is {prediction.formatted_price}."
    if prediction.relative_position == "below_typical_range":
        return base + " This is below the typical range in the training data."
    if prediction.relative_position == "above_typical_range":
        return base + " This is above the typical range in the training data."
    return base + " This is within the typical range in the training data."


def interpret_prediction(
    user_query: str,
    features: ExtractedFeatures,
    missing_fields: list[str],
    prediction: PredictionSummary,
    model_name: str,
    assumptions: list[str],
    version: str = "final",
) -> Stage2Response:
    needs_user_input = len(missing_fields) > 0

    try:
        prompt = build_interpretation_prompt(
            user_query=user_query,
            features=features,
            prediction=prediction,
            version=version,
        )

        interpretation = call_gemini_text(prompt)

        log_prompt_result({
            "stage": "stage2",
            "prompt_version": version,
            "input_query": user_query,
            "validated": True,
            "parsed_output": {
                "interpretation": interpretation,
                "prediction": prediction.model_dump(),
            },
        })

    except Exception as e:
        interpretation = fallback_interpretation(prediction)

        log_prompt_result({
            "stage": "stage2",
            "prompt_version": version,
            "input_query": user_query,
            "validated": False,
            "error": str(e),
            "parsed_output": {
                "interpretation": interpretation,
                "prediction": prediction.model_dump(),
            },
        })

        assumptions = assumptions + [f"Interpretation fallback used: {str(e)}"]

    return Stage2Response(
        features=features,
        missing_fields=missing_fields,
        needs_user_input=needs_user_input,
        prediction=prediction,
        interpretation=interpretation,
        assumptions=assumptions,
        interpretation_prompt_version=version,
        model_name=model_name,
    )