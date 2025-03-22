import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import json
import os

class PlantEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PlantEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)

class PlantRecommendationModel(nn.Module):
    def __init__(self, climate_features, preference_features, hidden_size, num_plants):
        super(PlantRecommendationModel, self).__init__()
        
        # Encoders for different inputs
        self.climate_encoder = PlantEncoder(climate_features, hidden_size)
        self.preference_encoder = PlantEncoder(preference_features, hidden_size)
        
        # Combined layers
        combined_size = (hidden_size // 2) * 2  # Size after concatenating both encoders
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_plants),
            nn.Sigmoid()  # Multi-label classification
        )
        
    def forward(self, climate_data, preference_data):
        climate_encoded = self.climate_encoder(climate_data)
        preference_encoded = self.preference_encoder(preference_data)
        
        # Concatenate features
        combined = torch.cat((climate_encoded, preference_encoded), dim=1)
        
        # Get plant scores
        plant_scores = self.classifier(combined)
        
        return plant_scores

class PlantRecommender:
    def __init__(self, model_path="models/plant_recommendation_model.pt", plant_data_path="data/plants.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load plant data
        with open(plant_data_path, 'r') as f:
            self.plants = json.load(f)
            
        # Initialize preprocessing components
        self.climate_scaler = None
        self.preference_encoder = None
        
        # Model parameters 
        climate_features = 5  # rainfall, temp, humidity, etc
        preference_features = 10  # encoded categorical features
        hidden_size = 128
        num_plants = len(self.plants)
        
        # Initialize model
        self.model = PlantRecommendationModel(
            climate_features,
            preference_features,
            hidden_size,
            num_plants
        ).to(self.device)
        
        # Load model weights if exists
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Loaded plant recommendation model")
        else:
            print("No pretrained model found. Training required.")
    
    def preprocess_inputs(self, climate_data, user_preferences):
        """Preprocess inputs for model inference"""
        # Here we would normally apply the same preprocessing as during training
        # For now, we'll mock this functionality
        climate_tensor = torch.tensor(list(climate_data.values()), dtype=torch.float32).unsqueeze(0)
        
        # One-hot encode categorical preferences
        pref_values = []
        for key in ["lightLevel", "waterFrequency", "experienceLevel", "plantPurpose"]:
            # This should be replaced with proper encoding that matches training
            pref_values.append(user_preferences.get(key, 0))
            
        # Add numerical features
        pref_values.append(user_preferences.get("spaceAvailable", 0) / 100.0)  # Normalize
        
        preference_tensor = torch.tensor(pref_values, dtype=torch.float32).unsqueeze(0)
        
        return climate_tensor.to(self.device), preference_tensor.to(self.device)
    
    def get_recommendations(self, climate_data, user_preferences, top_k=5):
        """Get top-k plant recommendations"""
        with torch.no_grad():
            climate_tensor, preference_tensor = self.preprocess_inputs(climate_data, user_preferences)
            plant_scores = self.model(climate_tensor, preference_tensor)
            
            # Get top-k indices
            _, top_indices = torch.topk(plant_scores[0], k=top_k)
            top_indices = top_indices.cpu().numpy()
            
            # Get recommendations
            recommendations = []
            for idx in top_indices:
                plant = self.plants[idx]
                recommendations.append({
                    "id": plant["id"],
                    "name": plant["name"],
                    "latinName": plant.get("latinName", ""),
                    "description": plant.get("description", ""),
                    "image": plant.get("image", ""),
                    "confidence": float(plant_scores[0][idx].item()),
                    "care": plant.get("care", {}),
                    "carbonSequestration": plant.get("carbonSequestration", "Medium")
                })
                
            return recommendations

    def train_model(self, training_data, epochs=50):
        """Train the recommendation model with training data"""
        # Convert training data to tensors
        climate_tensors = []
        preference_tensors = []
        labels = []
        
        for sample in training_data:
            climate_tensors.append(torch.tensor(sample["climate_features"], dtype=torch.float32))
            preference_tensors.append(torch.tensor(sample["preference_features"], dtype=torch.float32))
            labels.append(torch.tensor(sample["plant_labels"], dtype=torch.float32))
            
        climate_data = torch.stack(climate_tensors)
        preference_data = torch.stack(preference_tensors)
        plant_labels = torch.stack(labels)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(climate_data, preference_data)
            loss = criterion(outputs, plant_labels)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Save model
        torch.save(self.model.state_dict(), "models/plant_recommendation_model.pt")
        print("Model training completed and saved")