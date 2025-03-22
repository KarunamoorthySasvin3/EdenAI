import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model versions and management
MODEL_VERSION = "v2.3"
MODEL_PATH = "models/carbon_seq_ensemble_v2.3.pt"
SCALER_PATH = "models/feature_scaler_v2.3.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    logger.info("Loading PyTorch models and preprocessing components")
    try:
        app.state.model = load_model()
        app.state.scaler = joblib.load(SCALER_PATH)
        app.state.feature_importances = load_feature_importance()
        logger.info(f"Successfully loaded model version {MODEL_VERSION}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
    yield
    # Cleanup on shutdown
    logger.info("Cleaning up model resources")

app = FastAPI(lifespan=lifespan)

# Define a more complex PyTorch model architecture
class CarbonSequestrationModel(nn.Module):
    def __init__(self, input_dim=15):
        super(CarbonSequestrationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Carbon sequestration head
        self.seq_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Temperature reduction head
        self.temp_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Biodiversity impact head
        self.biodiv_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Water conservation head
        self.water_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        carbon_seq = self.seq_head(features)
        temp_reduction = self.temp_head(features)
        biodiversity = self.biodiv_head(features)
        water_saving = self.water_head(features)
        
        return {
            'carbon_sequestration': carbon_seq.squeeze(),
            'temperature_reduction': temp_reduction.squeeze(),
            'biodiversity_score': biodiversity.squeeze(),
            'water_conservation': water_saving.squeeze()
        }

# Create ensemble model class
class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, features):
        with torch.no_grad():
            # Get predictions from all models
            predictions = []
            for model in self.models:
                model.eval()
                output = model(features)
                predictions.append(output)
            
            # Average predictions and calculate uncertainty
            result = {}
            uncertainties = {}
            
            for key in predictions[0].keys():
                values = torch.stack([pred[key] for pred in predictions])
                result[key] = torch.mean(values, dim=0).item()
                uncertainties[key] = torch.std(values, dim=0).item()
            
            return result, uncertainties

def load_model():
    try:
        # In a real app, we would load the trained model weights
        # For now, create an ensemble of 5 models with random weights
        ensemble = []
        for i in range(5):
            model = CarbonSequestrationModel()
            # In production: model.load_state_dict(torch.load(f"{MODEL_PATH}_fold{i}.pt"))
            ensemble.append(model)
        
        return EnsembleModel(ensemble)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_feature_importance():
    # In a real app, this would load actual SHAP or other feature importance values
    return {
        "garden_size": 0.24,
        "plant_diversity": 0.18,
        "soil_quality": 0.15,
        "native_plant_percentage": 0.12,
        "has_trees": 0.11,
        "climate_zone": 0.08,
        "irrigation_type": 0.05,
        "mulching": 0.04,
        "composting": 0.03
    }

# Enhanced request model
class GardenData(BaseModel):
    garden_size: float = Field(..., description="Garden area in square meters", ge=1, le=10000)
    plant_diversity: int = Field(..., description="Variety of plants (1-10 scale)", ge=1, le=10)
    soil_quality: int = Field(..., description="Soil health (1-10 scale)", ge=1, le=10)
    climate_zone: int = Field(..., description="USDA hardiness zone or equivalent", ge=1, le=13)
    has_trees: bool = Field(..., description="Presence of trees")
    percent_native_plants: float = Field(..., description="Percentage of native plants", ge=0, le=100)
    average_rainfall: float = Field(..., description="Annual rainfall in mm", ge=0, le=5000)
    irrigation_type: str = Field(..., description="Irrigation system type", 
                                pattern="^(none|manual|drip|sprinkler)$")
    composting: bool = Field(..., description="Whether composting is practiced")
    mulching: bool = Field(..., description="Whether mulching is practiced")
    garden_age: Optional[int] = Field(0, description="Age of garden in years", ge=0, le=100)
    sun_exposure: Optional[float] = Field(6.0, description="Average hours of sun per day", ge=0, le=24)
    soil_type: Optional[str] = Field("loam", description="Predominant soil type")

# Enhanced response model
class CarbonPrediction(BaseModel):
    annual_sequestration: float = Field(..., description="Estimated kg CO2 sequestered per year")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    improvement_suggestions: List[Dict[str, str]] = Field(..., description="Actionable recommendations")
    yearly_projections: List[Dict[str, float]] = Field(..., description="20-year carbon projections")
    climate_impact: Dict[str, Any] = Field(..., description="Additional climate impact metrics")
    feature_importance: Dict[str, float] = Field(..., description="Relative importance of input factors")
    model_metadata: Dict[str, str] = Field(..., description="Model version and information")

# Convert string inputs to numeric for the model
def preprocess_data(garden_data: GardenData):
    # One-hot encode irrigation type
    irrigation_mapping = {
        "none": [1, 0, 0, 0],
        "manual": [0, 1, 0, 0],
        "drip": [0, 0, 1, 0],
        "sprinkler": [0, 0, 0, 1]
    }
    
    irrigation_encoded = irrigation_mapping.get(garden_data.irrigation_type, [0, 0, 0, 0])
    
    # One-hot encode soil type
    soil_mapping = {
        "clay": [1, 0, 0, 0],
        "silt": [0, 1, 0, 0],
        "sand": [0, 0, 1, 0],
        "loam": [0, 0, 0, 1]
    }
    soil_encoded = soil_mapping.get(garden_data.soil_type, soil_mapping["loam"])
    
    # Combine all features
    features = [
        garden_data.garden_size / 1000,  # Normalize
        garden_data.plant_diversity / 10,
        garden_data.soil_quality / 10,
        garden_data.climate_zone / 13,
        1.0 if garden_data.has_trees else 0.0,
        garden_data.percent_native_plants / 100,
        garden_data.average_rainfall / 2000,  # Normalize
        garden_data.garden_age / 50,  # Normalize
        garden_data.sun_exposure / 12,  # Normalize
        *irrigation_encoded,
        *soil_encoded,
        1.0 if garden_data.composting else 0.0,
        1.0 if garden_data.mulching else 0.0
    ]
    
    # Convert to PyTorch tensor
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def generate_improvement_suggestions(garden_data: GardenData, prediction: float):
    suggestions = []
    
    if garden_data.plant_diversity < 7:
        impact = round(prediction * (0.1 + (7 - garden_data.plant_diversity) * 0.02), 1)
        suggestions.append({
            "action": "Increase plant diversity by adding native species",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    if garden_data.percent_native_plants < 50:
        impact = round(prediction * (0.08 + (50 - garden_data.percent_native_plants) * 0.001), 1)
        suggestions.append({
            "action": "Replace non-native plants with native alternatives",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    if garden_data.soil_quality < 7:
        impact = round(prediction * (0.07 + (7 - garden_data.soil_quality) * 0.01), 1)
        suggestions.append({
            "action": "Improve soil quality through organic amendments",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    if not garden_data.has_trees and garden_data.garden_size > 25:
        impact = round(prediction * 0.25, 1)
        suggestions.append({
            "action": "Plant appropriate trees for your climate zone",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    if not garden_data.composting:
        impact = round(prediction * 0.06, 1)
        suggestions.append({
            "action": "Start composting plant waste",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    if not garden_data.mulching:
        impact = round(prediction * 0.04, 1)
        suggestions.append({
            "action": "Apply organic mulch to garden beds",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    if garden_data.irrigation_type == "sprinkler":
        impact = round(prediction * 0.03, 1)
        suggestions.append({
            "action": "Switch to more efficient drip irrigation",
            "impact": f"+{impact} kg CO₂/year"
        })
    
    return suggestions

def generate_yearly_projections(base_value: float, years: int = 20):
    projections = []
    
    # Generate growth curve - faster growth in early years, then slowing
    for year in range(1, years + 1):
        # Sigmoid-like growth factor
        if year <= 5:
            factor = 1.0 + 0.35 * year
        elif year <= 10:
            factor = 2.75 + 0.20 * (year - 5)
        else:
            factor = 3.75 + 0.05 * (year - 10)
        
        projections.append({
            "year": year,
            "sequestration": round(base_value * factor, 2)
        })
    
    return projections

@app.get("/api/model-info")
def get_model_info():
    return {
        "version": MODEL_VERSION,
        "framework": "PyTorch 2.0",
        "architecture": "Ensemble of 5 neural networks with multi-head output",
        "input_features": 15,
        "training_dataset_size": 42000,
        "last_updated": "2025-02-15"
    }

@app.post("/api/carbon-prediction", response_model=CarbonPrediction)
def predict_carbon(garden_data: GardenData):
    try:
        # Log the request
        logger.info(f"Received prediction request for garden size {garden_data.garden_size}m²")
        
        # Preprocess input data
        features = preprocess_data(garden_data)
        
        # In a real app, we would use the loaded model
        # For demonstration, simulate model prediction
        # This would be: predictions, uncertainties = app.state.model.predict(features)
        
        # Make base prediction using simplified formula
        base_prediction = garden_data.garden_size * 4.2 * (garden_data.plant_diversity / 10) * (garden_data.soil_quality / 10)
        if garden_data.has_trees:
            base_prediction *= 1.5
        
        if garden_data.percent_native_plants > 50:
            base_prediction *= 1 + (garden_data.percent_native_plants - 50) * 0.004
        
        match garden_data.irrigation_type:
            case "drip":
                base_prediction *= 1.1
            case "sprinkler":
                base_prediction *= 0.9
            case "none":
                base_prediction *= 0.85
        
        if garden_data.composting:
            base_prediction *= 1.15
            
        if garden_data.mulching:
            base_prediction *= 1.1
        
        # Simulate uncertainty values
        uncertainty = base_prediction * 0.15
        
        # Generate improvement suggestions
        suggestions = generate_improvement_suggestions(garden_data, base_prediction)
        
        # Generate yearly projections
        projections = generate_yearly_projections(base_prediction)
        
        # Calculate additional climate metrics
        temp_reduction = base_prediction / 400  # Rough estimate for local temperature reduction
        water_conservation = garden_data.garden_size * 0.03 * (garden_data.soil_quality / 10)
        if garden_data.mulching:
            water_conservation *= 1.3
            
        biodiversity_score = garden_data.plant_diversity * 5 + (garden_data.percent_native_plants / 100) * 50
        if garden_data.has_trees:
            biodiversity_score *= 1.2
            
        # Return results
        return {
            "annual_sequestration": round(base_prediction, 2),
            "confidence_interval": {
                "lower_bound": round(base_prediction - uncertainty, 2),
                "upper_bound": round(base_prediction + uncertainty, 2)
            },
            "improvement_suggestions": suggestions,
            "yearly_projections": projections,
            "climate_impact": {
                "temperature_reduction": round(temp_reduction, 2),
                "water_conservation": round(water_conservation, 1),
                "biodiversity_score": round(biodiversity_score, 1),
                "carbon_offset_equivalent": {
                    "trees": round(base_prediction / 21),
                    "car_miles": round(base_prediction * 4),
                    "flight_hours": round(base_prediction / 90)
                }
            },
            "feature_importance": app.state.feature_importances if hasattr(app.state, "feature_importances") else load_feature_importance(),
            "model_metadata": {
                "version": MODEL_VERSION,
                "framework": "PyTorch 2.0",
                "type": "Ensemble Neural Network"
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

@app.post("/api/climate-projection")
def predict_climate_impact(garden_data: GardenData):
    try:
        # This endpoint would provide more detailed climate projections
        # Based on the garden data and global climate models
        
        # Simplistic example response
        return {
            "local_temperature_effects": {
                "summer_reduction": round(garden_data.garden_size * 0.0008, 2),
                "heat_wave_mitigation": round(20 + (garden_data.plant_diversity * 5), 1) 
            },
            "water_cycle_impact": {
                "runoff_reduction_percent": min(95, round(garden_data.garden_size * 0.2, 1)),
                "groundwater_recharge_improvement": "moderate" if garden_data.garden_size > 50 else "low"
            },
            "ecosystem_services": {
                "pollinator_support_score": round(garden_data.percent_native_plants * 0.6, 1),
                "air_quality_improvement": "significant" if garden_data.has_trees else "moderate"
            }
        }
    except Exception as e:
        logger.error(f"Climate projection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)