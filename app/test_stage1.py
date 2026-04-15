from app.services.feature_extractor import extract_features


TEST_QUERIES = [
    "A 3-bedroom 2-story house with 2 bathrooms and a 2-car garage in NAmes.",
    "Modern luxury home with a large lot and finished basement.",
    "Small ranch-style house, 2 bedrooms, 1 bathroom, older home.",
    "A 2200 sqft house built in 2010 with a basement and 3-car garage.",
    "3-bed in a good neighborhood.",
]


def main():
    for version in ["v1", "v2"]:
        print(f"\n--- Testing prompt {version} ---")
        for query in TEST_QUERIES:
            result = extract_features(query, version=version)
            print("\nQuery:", query)
            print(result)


if __name__ == "__main__":
    main()