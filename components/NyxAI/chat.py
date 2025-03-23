import json
import torch
from model import Reccomendations
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = Reccomendations(data['input_size'], data['hidden_size'], data['output_size']).to(device)
model.load_state_dict(model_state)
model.eval()

text = ""
while text != "quit":
    text = input("Enter: ") #replace with imput in html
    text = tokenize(text)
    message = bag_of_words(text, words)
    message = message.reshape(1, message.shape[0])
    message = torch.from_numpy(message)

    output = model(message)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    for intent in intents['intents']:
        if tag == intent["tag"]:
            print(intent["responses"])