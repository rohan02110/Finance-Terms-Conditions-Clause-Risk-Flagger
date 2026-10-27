"""
Export trained PyTorch ANN weights, TF-IDF vocabulary, and Knowledge Base
to lightweight JSON for serverless Vercel execution and browser inference.
"""

import os
import json
import joblib
import torch
import numpy as np

def export_model():
    print("Loading PyTorch model checkpoint and TF-IDF vectorizer...")
    ckpt = torch.load("models/best_ann_model.pt", weights_only=True)
    state = ckpt["model_state_dict"]
    vec = joblib.load("models/tfidf_vectorizer.joblib")

    # Extract weights as float lists
    fc1_w = state["fc1.weight"].cpu().numpy().tolist()  # (256, 5000)
    fc1_b = state["fc1.bias"].cpu().numpy().tolist()    # (256,)
    fc2_w = state["fc2.weight"].cpu().numpy().tolist()  # (64, 256)
    fc2_b = state["fc2.bias"].cpu().numpy().tolist()    # (64,)
    fc_out_w = state["fc_out.weight"].cpu().numpy().tolist() # (1, 64)
    fc_out_b = state["fc_out.bias"].cpu().numpy().tolist()   # (1,)

    vocab = {str(k): int(v) for k, v in vec.vocabulary_.items()}
    idfs = [float(x) for x in vec.idf_]

    # Load Knowledge Base if available
    kb_entries = []
    kb_path = "data/knowledge_base/rules.csv"
    if os.path.exists(kb_path):
        import pandas as pd
        df = pd.read_csv(kb_path)
        kb_entries = df.fillna("").to_dict(orient="records")

    export_payload = {
        "metadata": {
            "model_type": "2-layer PyTorch Feed-Forward ANN (MLP)",
            "input_dim": 5000,
            "hidden_dim1": 256,
            "hidden_dim2": 64,
            "default_threshold": 0.05,
            "recall_tuned_threshold": 0.024,
            "val_pr_auc": ckpt.get("best_val_pr_auc", 0.766)
        },
        "vocab": vocab,
        "idf": idfs,
        "weights": {
            "fc1_w": fc1_w,
            "fc1_b": fc1_b,
            "fc2_w": fc2_w,
            "fc2_b": fc2_b,
            "fc_out_w": fc_out_w,
            "fc_out_b": fc_out_b
        },
        "knowledge_base": kb_entries
    }

    os.makedirs("public", exist_ok=True)
    out_file = "public/model_data.json"
    print(f"Writing export to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f)

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"Export completed: {out_file} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    export_model()
