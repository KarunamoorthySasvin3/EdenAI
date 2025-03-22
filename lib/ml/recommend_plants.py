import sys
import json
import torch
from .plant_recommendation_model import PlantRecommender

def main():
    """
    Entry point for plant recommendations
    Usage: python -m lib.ml.recommend_plants '{"climate": {...}, "preferences": {...}}'
    """
    # Parse input from command line
    input_json = sys.argv[1]
    input_data = json.loads(input_json)
    
    # Extract data
    climate_data = input_data.get("climate", {})
    preferences = input_data.get("preferences", {})
    
    # Initialize the PyTorch model
    recommender = PlantRecommender(
        model_path="models/plant_recommendation_model.pt", 
        plant_data_path="data/plants.json"
    )
    
    # Get recommendations
    recommendations = recommender.get_recommendations(climate_data, preferences, top_k=5)
    
    # Return as JSON
    print(json.dumps(recommendations))

if __name__ == "__main__":
    main()