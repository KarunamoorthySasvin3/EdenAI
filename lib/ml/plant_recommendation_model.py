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
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # For multi-label scores
        )

    def forward(self, x):
        return self.model(x)

class PlantRecommender:
    def __init__(self, model_path="models/plant_recommendation_model.pt", plant_data_path="data/plants.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load plant data
        with open(plant_data_path, 'r') as f:
            self.plants = json.load(f)
            
        # Initialize preprocessing components
        self.climate_scaler = StandardScaler()
        self.preference_encoder = OneHotEncoder(handle_unknown='ignore')
        
        # Model parameters 
        climate_features = 5  # rainfall, temp, humidity, etc
        preference_features = 10  # encoded categorical features
        hidden_size = 128
        num_plants = len(self.plants)
        
        # Initialize model
        self.model = PlantRecommendationModel(
            climate_features + preference_features,
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
    
    def preprocess_data(self, plants):
        # Separate features
        climate_features = [[plant['rainfall'], plant['temperature'], plant['humidity'], plant['sunlight_hours'], plant['soil_ph']] for plant in plants]
        preference_features = [[plant['plant_type'], plant['water_needs'], plant['fertilizer_needs'], plant['light_level'], plant['flower_color'], plant['foliage_color'], plant['growth_rate'], plant['cold_hardiness'], plant['drought_tolerance'], plant['pest_resistance']] for plant in plants]

        # Scale climate features
        climate_scaled = self.climate_scaler.fit_transform(climate_features)

        # Encode preference features
        preference_encoded = self.preference_encoder.fit_transform(preference_features).toarray()

        return climate_scaled, preference_encoded

    def get_recommendations(self, climate_data, user_preferences, top_k=5):
        """Get top-k plant recommendations"""
        with torch.no_grad():
            climate_tensor, preference_tensor = self.preprocess_inputs(climate_data, user_preferences)
            # Combine inputs along feature dimension
            inputs = torch.cat([climate_tensor, preference_tensor], dim=1)
            plant_scores = self.model(inputs)  # updated to use concatenated inputs
            # Get top-k indices
            _, top_indices = torch.topk(plant_scores[0], k=top_k)
            top_indices = top_indices.cpu().numpy()
            
            recommendations = []
            for idx in top_indices:
                recommendations.append({
                    "id": self.plants[idx]["id"],
                    "name": self.plants[idx]["name"],
                    "latinName": self.plants[idx].get("latinName", ""),
                    "description": self.plants[idx].get("description", ""),
                    "image": self.plants[idx].get("image", ""),
                    "confidence": float(plant_scores[0][idx].item()),
                    "care": self.plants[idx].get("care", {}),
                    "carbonSequestration": self.plants[idx].get("carbonSequestration", "Medium")
                })
                
            return recommendations

    def train_model(self, training_data, epochs=50):
        """Train the recommendation model with training data"""
        # Preprocess data
        climate_scaled, preference_encoded = self.preprocess_data(training_data)

        # Combine features
        X = np.concatenate((climate_scaled, preference_encoded), axis=1)
        y = np.array([plant['plant_id'] for plant in training_data])  # Assuming each plant has a unique ID

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # Save model
        torch.save(self.model.state_dict(), "models/plant_recommendation_model.pt")
        print("Model training completed and saved")
    
    def recommend_plants(self, climate_data, preferences):
        # Process and convert input
        climate_features = [
            climate_data.get("rainfall", 0),
            climate_data.get("temperature", 0),
            climate_data.get("humidity", 0),
            climate_data.get("sunlightHours", 0),
            climate_data.get("zone", 0)
        ]
        
        # Extract preference features
        goals = preferences.get("goals", [])
        space = preferences.get("spaceAvailable", 0)
        preference_features = [
            1 if "food" in goals else 0,
            1 if "lowMaintenance" in goals else 0,
            1 if "pollinator" in goals else 0,
            1 if "medicinal" in goals else 0
        ]
        
        combined_features = climate_features + preference_features
        input_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            scores = self.model(input_tensor)
            _, top_indices = torch.topk(scores, k=5)
            top_indices = top_indices.cpu().numpy()
            
            recommendations = []
            for idx in top_indices[0]:
                plant = self.plants[idx]
                recommendations.append({
                    "id": plant["id"],
                    "name": plant["name"],
                    "latinName": plant.get("latinName", ""),
                    "description": plant.get("description", ""),
                    "image": plant.get("image", ""),
                    "confidence": float(scores[0][idx].item()),
                    "care": plant.get("care", {}),
                    "carbonSequestration": plant.get("carbonSequestration", "Medium")
                })
                
            return recommendations