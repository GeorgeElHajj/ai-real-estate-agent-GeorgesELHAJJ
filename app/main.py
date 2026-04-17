from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schemas.api import (
    QueryRequest,
    IncompleteResponse,
    CompleteResponse,
    PredictFromFeaturesRequest,
)
from app.schemas.extraction import ExtractedFeatures
from app.schemas.image import ImageGenerationRequest, ImageGenerationResponse
from app.services.feature_extractor import extract_features
from app.services.predictor import PredictorService
from app.services.interpreter import interpret_prediction
from app.services.image_generator import generate_house_image


app = FastAPI(
    title="AI Real Estate Agent API",
    version="1.0.0",
    description="LLM extraction + ML prediction + LLM interpretation + image generation"
)

predictor_service = PredictorService()


@app.get("/")
def root():
    return {"message": "AI Real Estate Agent API is running."}


@app.post("/predict", response_model=None)
def predict_price(request: QueryRequest):
    user_query = request.query.strip()

    if not user_query:
        return JSONResponse(
            status_code=400,
            content={"error": "Query must not be empty."}
        )

    print("QUERY RECEIVED:", user_query)

    stage1_result = extract_features(user_query, version="final")
    print("STAGE1 RESULT:", stage1_result)

    features_dict = stage1_result["features"]
    extracted_fields = stage1_result["extracted_fields"]
    missing_fields = stage1_result["missing_fields"]
    assumptions = stage1_result["assumptions"]
    prompt_version = stage1_result["prompt_version"]

    features_obj = ExtractedFeatures(**features_dict)

    if missing_fields:
        response = IncompleteResponse(
            status="incomplete",
            message="Please provide the missing fields before prediction.",
            extracted_features=features_obj,
            extracted_fields=extracted_fields,
            missing_fields=missing_fields,
            assumptions=assumptions,
            prompt_version=prompt_version,
        )
        return response.model_dump()

    prediction = predictor_service.predict(features_obj)

    stage2_response = interpret_prediction(
        user_query=user_query,
        features=features_obj,
        missing_fields=missing_fields,
        prediction=prediction,
        model_name=predictor_service.get_model_name(),
        assumptions=assumptions,
        version="final",
    )

    response = CompleteResponse(
        status="complete",
        message="Prediction completed successfully.",
        extracted_features=features_obj,
        extracted_fields=extracted_fields,
        missing_fields=missing_fields,
        assumptions=stage2_response.assumptions,
        prediction=stage2_response.prediction,
        interpretation=stage2_response.interpretation,
        model_name=stage2_response.model_name,
        extraction_prompt_version=prompt_version,
        interpretation_prompt_version=stage2_response.interpretation_prompt_version,
    )

    return response.model_dump()


@app.post("/predict_from_features", response_model=None)
def predict_from_features(request: PredictFromFeaturesRequest):
    features_obj = request.features
    query = request.query

    prediction = predictor_service.predict(features_obj)

    stage2_response = interpret_prediction(
        user_query=query,
        features=features_obj,
        missing_fields=[],
        prediction=prediction,
        model_name=predictor_service.get_model_name(),
        assumptions=[],
        version="final",
    )

    response = CompleteResponse(
        status="complete",
        message="Prediction completed successfully.",
        extracted_features=features_obj,
        extracted_fields=[
            k for k, v in features_obj.model_dump().items() if v is not None
        ],
        missing_fields=[],
        assumptions=stage2_response.assumptions,
        prediction=stage2_response.prediction,
        interpretation=stage2_response.interpretation,
        model_name=stage2_response.model_name,
        extraction_prompt_version="final",
        interpretation_prompt_version=stage2_response.interpretation_prompt_version,
    )

    return response.model_dump()


@app.post("/generate-image", response_model=ImageGenerationResponse)
def generate_image(request: ImageGenerationRequest):
    return generate_house_image(request.query, request.features)