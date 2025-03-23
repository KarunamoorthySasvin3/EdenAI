import sys
import json
import random
import torch
import os
from pathlib import Path

# Determine the correct paths based on the project structure
BASE_DIR = Path(__file__).resolve().parent.parent
NICKSAI_DIR = BASE_DIR / "components" / "NicksAI"

# Add the project base directory to sys.path so that packages can be resolved
sys.path.insert(0, str(BASE_DIR))

# Import the NicksAI components using the absolute package path
from components.NicksAI.chat import NeuralNet
from components.NicksAI.nltk_utils import bag_of_words, tokenize

def get_response(message):
    # Set working directory to NicksAI directory for file loading
    os.chdir(str(NICKSAI_DIR))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    
    FILE = "data.pth"
    data = torch.load(FILE)
    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']
    
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Process the input message
    text = tokenize(message)
    X = bag_of_words(text, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # Return the appropriate response based on the prediction
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    else:
        return "I'm not sure I understand. Could you please rephrase that?"

if __name__ == "__main__":
    # Accept input as a JSON string argument
    input_json = json.loads(sys.argv[1])
    message = input_json.get("message", "")
    
    response = {
        "response": get_response(message)
    }
    
    # Return JSON response
    print(json.dumps(response))