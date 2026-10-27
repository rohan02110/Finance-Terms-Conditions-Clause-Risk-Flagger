"""
Training Pipeline for Clause Risk ANN.

Key Design Decisions & Educational Notes:
1. Feature Extraction:
   - TF-IDF vectorizer is fitted ONLY on the training split.
   - Validation and test splits are transformed using the fitted training vocabulary.
   - This ensures zero data leakage.

2. Class Imbalance Handling:
   - In our training data: ~4,902 Fair (Class 0) vs 630 Unfair (Class 1).
   - Imbalance ratio = 4,902 / 630 ≈ 7.78.
   - We pass `pos_weight = torch.tensor([7.78])` to `nn.BCEWithLogitsLoss`.
   - WHY: Without pos_weight, the loss function treats an error on class 0 and class 1
     identically. The model could easily achieve ~89% accuracy by predicting 0 almost always,
     while missing predatory clauses. `pos_weight` scales the loss for false negatives by ~7.78x,
     penalizing missed risky clauses and forcing the optimizer to prioritize recall.

3. Optimization & Early Stopping:
   - Adam optimizer with lr=1e-3 and weight decay for slight L2 regularization.
   - Monitored metric: Validation PR-AUC (Average Precision score).
   - PR-AUC directly measures ranking quality on the imbalanced positive class.
   - Early stopping stops training when validation PR-AUC stops improving for `patience` epochs.

4. Multi-Seed Robustness:
   - Trains across 3 fixed random seeds (42, 123, 999).
   - Reports mean and spread of validation metrics across seeds.
   - Saves the best overall checkpoint.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from src.features import fit_and_save_vectorizer
from src.model import ClauseRiskANN


def set_seed(seed: int):
    """Ensure deterministic execution across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_seed(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    seed: int,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5
):
    """Train the ANN for one seed with early stopping on validation PR-AUC."""
    set_seed(seed)

    # Convert sparse / numpy matrices to PyTorch float tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_np.shape[1]
    model = ClauseRiskANN(input_dim=input_dim)

    # Calculate class imbalance weighting: neg_count / pos_count
    n_pos = (y_train_np == 1).sum()
    n_neg = (y_train_np == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_f1": []
    }

    best_pr_auc = -1.0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_total = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * len(batch_x)

        train_loss_avg = train_loss_total / len(train_dataset)

        # 2. Validation Phase
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_preds = (val_probs >= 0.5).astype(int)

        val_pr_auc = average_precision_score(y_val_np, val_probs)
        val_f1 = f1_score(y_val_np, val_preds, pos_label=1, zero_division=0)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_f1"].append(val_f1)

        print(f"Seed {seed} | Epoch {epoch:02d} | Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_f1:.4f}")

        # Check early stopping condition
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    return model, history, best_pr_auc


def run_training():
    print("=== Starting Phase 3 ANN Training Pipeline ===")
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    X_train_text = train_df["text"].fillna("")
    y_train = train_df["label"].values

    X_val_text = val_df["text"].fillna("")
    y_val = val_df["label"].values

    # 2. Fit TF-IDF Vectorizer Strictly on Train Split
    vectorizer = fit_and_save_vectorizer(X_train_text, "models/tfidf_vectorizer.joblib")
    X_train_features = vectorizer.transform(X_train_text).toarray()
    X_val_features = vectorizer.transform(X_val_text).toarray()

    # 3. Multi-Seed Training Loop
    seeds = [42, 123, 999]
    seed_results = []
    best_overall_model = None
    best_overall_pr_auc = -1.0
    best_history = None

    for seed in seeds:
        print(f"\n--- Training ANN with Seed {seed} ---")
        model, history, best_pr_auc = train_single_seed(
            X_train_features, y_train,
            X_val_features, y_val,
            seed=seed, epochs=25, batch_size=64, lr=1e-3, patience=5
        )
        seed_results.append(best_pr_auc)

        if best_pr_auc > best_overall_pr_auc:
            best_overall_pr_auc = best_pr_auc
            best_overall_model = model
            best_history = history

    # 4. Save Best Model Checkpoint
    checkpoint_path = "models/best_ann_model.pt"
    torch.save({
        "model_state_dict": best_overall_model.state_dict(),
        "input_dim": X_train_features.shape[1],
        "best_val_pr_auc": best_overall_pr_auc,
        "seeds_evaluated": seeds,
        "seed_scores": seed_results
    }, checkpoint_path)
    print(f"\nBest model saved to {checkpoint_path} with Val PR-AUC: {best_overall_pr_auc:.4f}")

    # 5. Save Training & Validation Loss Curves Plot
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(best_history["epoch"], best_history["train_loss"], label="Train Loss", color="royalblue", marker="o")
    plt.plot(best_history["epoch"], best_history["val_loss"], label="Val Loss", color="crimson", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Weighted BCE)")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(best_history["epoch"], best_history["val_pr_auc"], label="Val PR-AUC", color="darkgreen", marker="^")
    plt.plot(best_history["epoch"], best_history["val_f1"], label="Val F1", color="purple", marker="d")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curve_path = "reports/figures/training_loss_curves.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {curve_path}")

    # 6. Print Multi-Seed Summary
    mean_pr_auc = np.mean(seed_results)
    std_pr_auc = np.std(seed_results)
    print(f"\n=== Multi-Seed Validation PR-AUC Summary ===")
    for s, score in zip(seeds, seed_results):
        print(f"Seed {s}: {score:.4f}")
    print(f"Mean PR-AUC: {mean_pr_auc:.4f} (+/- {std_pr_auc:.4f})")


if __name__ == "__main__":
    run_training()
