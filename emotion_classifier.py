# emotion_classifier.py
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionClassifier:
    def __init__(self, model_path="models/emotion_detector_final", threshold=0.45):
        self.model_path = model_path
        self.threshold = threshold

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # Load model (safetensors)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True
        )

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Load real GoEmotions label names
        labels_file = os.path.join(model_path, "label_names.txt")
        with open(labels_file, "r") as f:
            self.label_names = [line.strip() for line in f.readlines()]

        print("Model loaded successfully with real GoEmotions labels!")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            idxs = np.where(p >= self.threshold)[0]
            labels = [self.label_names[j] for j in idxs]

            if not labels:  # fallback: top-3
                top3 = p.argsort()[-3:][::-1]
                labels = [self.label_names[j] for j in top3]

            results.append({"text": texts[i], "labels": labels})

        return results


if __name__ == "__main__":
    clf = EmotionClassifier()
    print("Model loaded successfully!")
