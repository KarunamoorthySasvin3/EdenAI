# filepath: train_recommender.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from lib.ml.plant_recommendation_model import PlantRecommender

# ... load your training dataset ...
training_data = []
recommender = PlantRecommender()
recommender.train_model(training_data, epochs=50)