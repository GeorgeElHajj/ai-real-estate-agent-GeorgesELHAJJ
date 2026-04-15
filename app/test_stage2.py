from app.schemas.extraction import ExtractedFeatures
from app.services.feature_extractor import extract_features
from app.services.predictor import PredictorService
from app.services.interpreter import interpret_prediction


TEST_QUERIES = [
    "A 3-bedroom 2-story house with 2 bathrooms and a 2-car garage in NAmes.",
    "Small ranch-style house, 2 bedrooms, 1 bathroom, older home.",
    "A 2200 sqft house built in 2010 with a basement and 3-car garage.",
]


def fill_demo_missing_values(features: dict) -> dict:
    demo_defaults = {
        "OverallQual": 6,
        "GrLivArea": 1800.0,
        "Neighborhood": "NAmes",
        "TotalBsmtSF": 900.0,
        "GarageCars": 2.0,
        "FullBath": 2.0,
        "LotArea": 8500.0,
        "BedroomAbvGr": 3,
        "HouseStyle": "1Story",
        "HouseAge": 15,
    }

    completed = features.copy()
    for key, value in demo_defaults.items():
        if completed.get(key) is None:
            completed[key] = value

    return completed


def main():
    predictor = PredictorService()

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print("QUERY:", query)

        stage1_result = extract_features(query, version="final")
        print("\nSTAGE 1 RESULT:")
        print(stage1_result)

        completed_features_dict = fill_demo_missing_values(stage1_result["features"])
        features_obj = ExtractedFeatures(**completed_features_dict)

        prediction = predictor.predict(features_obj)

        stage2_response = interpret_prediction(
            user_query=query,
            features=features_obj,
            missing_fields=stage1_result["missing_fields"],
            prediction=prediction,
            model_name=predictor.get_model_name(),
            assumptions=stage1_result["assumptions"],
            version="final",
        )

        print("\nSTAGE 2 RESPONSE:")
        print(stage2_response.model_dump())


if __name__ == "__main__":
    main()