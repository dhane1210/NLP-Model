# app.py - Updated Flask API
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load tokenizer, model, and label encoder
tokenizer = AutoTokenizer.from_pretrained("mental_bert_model")
model = AutoModelForSequenceClassification.from_pretrained("mental_bert_model")

with open("mental_bert_model/mental_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Setup device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_condition(text: str):
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    # Decode label
    return label_encoder.inverse_transform([predicted.item()])[0], confidence.item()

# Prediction API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        prediction, confidence = predict_condition(text)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
