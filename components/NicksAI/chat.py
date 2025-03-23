import random
import json
import torch
from chatbot import Chatbot
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = Chatbot(data['input_size'], data['hidden_size'], data['output_size']).to(device)
model.load_state_dict(model_state)
model.eval()

text = ""
while text != "quit":
    text = input("Enter: ") #replace with imput in html
    text = tokenize(text)
    X = bag_of_words(text, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(random.choice(intent['responses'])) #output
    else:
        print("Sorry, I dont understand.") #output