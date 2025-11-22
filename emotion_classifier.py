# emotion_classifier.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionClassifier:
    def __init__(self, model_path="OVERLOOKHOTEL13/emotion-detector-final", threshold=0.45):
        """
        Load model & tokenizer from HuggingFace Hub
        """
        print(f"Loading model from HuggingFace repo: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_auth_token=None  # or HF token if private
        )

        # Load model using safetensors
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            use_safetensors=True,
            use_auth_token=None  # or HF token if private
        )

        # Set evaluation mode
        self.model.eval()

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Load real GoEmotions labels (you uploaded this file!)
        try:
            with open("label_names.txt", "r") as f:
                self.label_names = [line.strip() for line in f.readlines()]
            print("Loaded real label names.")
        except:
            print("⚠ WARNING: label_names.txt not found, using placeholder labels.")
            self.label_names = [f"label_{i}" for i in range(self.model.config.num_labels)]

        self.threshold = threshold

    def predict(self, texts):
        """
        Predict emotions for a single text or list of texts.
        Returns:
            [{ "text": ..., "labels": [...] }]
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Predict
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for text, p in zip(texts, probs):
            # Labels above threshold
            idxs = np.where(p >= self.threshold)[0]
            labels = [self.label_names[i] for i in idxs]

            # If none above threshold → pick top 3
            if not labels:
                top3 = p.argsort()[-3:][::-1]
                labels = [self.label_names[i] for i in top3]

            results.append({
                "text": text,
                "labels": labels
            })

        return results
