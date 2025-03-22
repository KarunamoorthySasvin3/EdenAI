# Create a PyTorch plant recommendation model
import torch
import torch.nn as nn

class PlantRecommendationModel(nn.Module):
    def __init__(self, input_features, hidden_size, output_size):
        super(PlantRecommendationModel, self).__init__()
        self.climate_encoder = nn.Linear(input_features, hidden_size)
        self.user_pref_encoder = nn.Linear(input_features, hidden_size)
        self.combined = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, climate_data, user_preferences):
        climate_encoded = torch.relu(self.climate_encoder(climate_data))
        prefs_encoded = torch.relu(self.user_pref_encoder(user_preferences))
        combined = torch.cat((climate_encoded, prefs_encoded), dim=1)
        hidden = torch.relu(self.combined(combined))
        return self.output(hidden)