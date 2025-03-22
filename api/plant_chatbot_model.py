import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from transformers import AutoTokenizer, AutoModel
import sqlite3

class PlantChatbotEncoder(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased"):
        super(PlantChatbotEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
    def forward(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get embeddings
        outputs = self.bert(**inputs)
        
        # Use [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings

class PlantKnowledgeRetriever(nn.Module):
    def __init__(self, hidden_size=768, knowledge_size=512):
        super(PlantKnowledgeRetriever, self).__init__()
        self.knowledge_projection = nn.Linear(hidden_size, knowledge_size)
        
    def forward(self, query_embedding, plant_embedding):
        # Project to knowledge space
        query_proj = self.knowledge_projection(query_embedding)
        plant_proj = self.knowledge_projection(plant_embedding)
        
        # Calculate relevance score
        relevance = F.cosine_similarity(query_proj, plant_proj)
        
        return relevance

class PlantResponseGenerator(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(PlantResponseGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, query_embedding, plant_knowledge):
        # Combine query and plant knowledge
        combined = torch.cat([query_embedding, plant_knowledge], dim=1)
        
        # Generate response embedding
        response_embedding = self.generator(combined)
        
        return response_embedding

class PlantChatbot:
    def __init__(self, model_dir="models/plant_chatbot/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load plant knowledge base
        self.plant_data = self._load_plant_knowledge()
        
        # Initialize component models
        self.encoder = PlantChatbotEncoder().to(self.device)
        self.retriever = PlantKnowledgeRetriever().to(self.device)
        self.generator = PlantResponseGenerator().to(self.device)
        
        # Load pre-computed plant embeddings
        self.plant_embeddings = self._load_or_compute_embeddings(model_dir)
        
        # Database connection
        self.db_path = "plant_chatbot.db"
        self._init_database()
    
    def _load_plant_knowledge(self):
        """Load plant knowledge from JSON file"""
        try:
            with open("data/plants_knowledge.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Create a minimal plant database if none exists
            default_data = {
                "general": {
                    "info": "Plants generally need light, water, and nutrients to grow.",
                    "care_tips": "Most plants benefit from regular watering and appropriate light conditions."
                }
            }
            return default_data
    
    def _load_or_compute_embeddings(self, model_dir):
        """Load or compute plant knowledge embeddings"""
        embedding_path = os.path.join(model_dir, "plant_embeddings.pt")
        
        try:
            return torch.load(embedding_path, map_location=self.device)
        except FileNotFoundError:
            # Compute embeddings for all plants
            embeddings = {}
            for plant_name, plant_info in self.plant_data.items():
                # Combine all plant information into a single text
                plant_text = f"{plant_name}: "
                for key, value in plant_info.items():
                    if isinstance(value, str):
                        plant_text += f"{key} - {value}. "
                    elif isinstance(value, list):
                        plant_text += f"{key} - {', '.join(value)}. "
                
                # Generate embedding
                with torch.no_grad():
                    embedding = self.encoder(plant_text)
                    embeddings[plant_name] = embedding
            
            # Save embeddings
            torch.save(embeddings, embedding_path)
            return embeddings
    
    def _init_database(self):
        """Initialize SQLite database for chat history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            plant_context TEXT,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_chat(self, user_id, message, response, plant_context=None):
        """Save chat to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO chat_history (user_id, plant_context, message, response) VALUES (?, ?, ?, ?)",
            (user_id, plant_context, message, response)
        )
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, user_id, plant_context=None):
        """Get chat history for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if plant_context:
            cursor.execute(
                "SELECT message, response FROM chat_history WHERE user_id = ? AND plant_context = ? ORDER BY timestamp",
                (user_id, plant_context)
            )
        else:
            cursor.execute(
                "SELECT message, response FROM chat_history WHERE user_id = ? ORDER BY timestamp",
                (user_id,)
            )
        
        history = cursor.fetchall()
        conn.close()
        
        # Format history
        formatted_history = []
        for msg, resp in history:
            formatted_history.append({"role": "user", "content": msg})
            formatted_history.append({"role": "bot", "content": resp})
        
        return formatted_history
    
    def generate_response(self, message, plant_name=None, history=None):
        """Generate response to user message"""
        with torch.no_grad():
            # Encode user message
            query_embedding = self.encoder(message)
            
            # Get plant embedding
            if plant_name and plant_name.lower() in self.plant_embeddings:
                plant_embedding = self.plant_embeddings[plant_name.lower()]
            else:
                # Use general plant knowledge
                plant_embedding = self.plant_embeddings.get("general", query_embedding)
            
            # Generate response (in a real implementation, this would use a more sophisticated approach)
            # For demonstration, we'll retrieve relevant knowledge and create a template-based response
            if plant_name:
                plant_data = self.plant_data.get(plant_name.lower(), self.plant_data.get("general", {}))
                
                if "water" in message.lower():
                    return f"{plant_name} typically needs {plant_data.get('watering', 'regular watering')}. Make sure to check the soil moisture before watering."
                elif "light" in message.lower():
                    return f"{plant_name} thrives in {plant_data.get('light', 'moderate light conditions')}. Adjust placement accordingly."
                elif "fertiliz" in message.lower():
                    return f"For {plant_name}, use {plant_data.get('fertilizer', 'a balanced fertilizer')} during the growing season."
                else:
                    return f"{plant_name} is {plant_data.get('description', 'a wonderful plant to grow')}. Is there something specific you'd like to know about caring for it?"
            else:
                # General response
                return "I can help with plant care advice. For more specific guidance, please specify which plant you're asking about."