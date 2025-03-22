import sys
import json
from api.plant_chatbot_model import PlantChatbot

def main():
    """
    Entry point for chat responses
    Usage: python -m api.chat_endpoint '{"userId": "...", "message": "...", "plant": "...", "history": [...]}'
    """
    # Parse input from command line
    input_json = sys.argv[1]
    input_data = json.loads(input_json)
    
    # Extract data
    user_id = input_data.get("userId", "anonymous")
    message = input_data.get("message", "")
    plant = input_data.get("plant")
    history = input_data.get("history", [])
    
    # Initialize the PyTorch chatbot
    chatbot = PlantChatbot()
    
    # Generate response
    response = chatbot.generate_response(message, plant_name=plant, history=history)
    
    # Save to history
    chatbot.save_chat(user_id, message, response, plant_context=plant)
    
    # Return as JSON
    print(json.dumps({"response": response}))

if __name__ == "__main__":
    main()