import torch
import torch.nn as nn
import joblib
import re
import csv
import os
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the Model Class 
class NetBinary(nn.Module):
    def __init__(self, input_dim):
        super(NetBinary, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class MedicalAgent:
    def __init__(self, model_path="userIntents.pth", vectorizer_path="vectorizer.pkl", dataset_path="dataset.csv"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.dataset_path = dataset_path
        
        # Load Vectorizer
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            print("Warning: Vectorizer not found. Agent will not work correctly until trained.")
            self.vectorizer = None

        # Load Model
        if os.path.exists(self.model_path) and self.vectorizer:
            input_dim = len(self.vectorizer.get_feature_names_out())
            self.model = NetBinary(input_dim)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            print("Warning: Model not found.")
            self.model = None

        self.medicines = [
            "Panadol Extra", "Aspirin", "Telfast", "Metformin",
            "Insulin Pen", "Tamoxifen", "Imatinib", "Orlistat",
            "Prozac", "Zoloft", "Amoxicillin", "Cough syrup", "Paracetamol", "Vitamin C"
        ]
        
        self.number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def predict_intent(self, prompt):
        if not self.model or not self.vectorizer:
            return "unknown", 0.0

        prompt_clean = self.clean_text(prompt)
        vec = self.vectorizer.transform([prompt_clean]).toarray()
        tensor = torch.tensor(vec, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()

        label = 'buy_medicine' if prob < 0.5 else 'add_to_cart'
        confidence = (1 - prob) if label == 'buy_medicine' else prob
        return label, confidence

    def parse_quantity(self, word):
        word = word.lower()
        if word.isdigit():
            return int(word)
        if word in self.number_words:
            return self.number_words[word]
        return None

    def extract_entities(self, prompt):
        words = prompt.split()
        quantity = None

        # 1. Look for quantity
        for w in words:
            q = self.parse_quantity(w)
            if q is not None:
                quantity = q
                break
        
        if quantity is None:
            quantity = 1

        # 2. Remove numbers for matching
        prompt_no_numbers = ' '.join([w for w in words if self.parse_quantity(w) is None])

        # 3. Fuzzy match medicine
        best_match, score = process.extractOne(prompt_no_numbers, self.medicines)
        medicine = best_match if score >= 70 else None

        return medicine, quantity

    def save_new_data(self, prompt, intent):

        
        file_exists = os.path.exists(self.dataset_path)
        
        with open(self.dataset_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            writer.writerow([f'"{prompt}"', f'"{intent}"'])
            
        print(f"Saved new training data: {prompt} -> {intent}")

    def process_request(self, prompt):
        intent, confidence = self.predict_intent(prompt)
        medicine, quantity = self.extract_entities(prompt)
        return {
            "intent": intent,
            "confidence": confidence,
            "medicine": medicine,
            "quantity": quantity
        }
