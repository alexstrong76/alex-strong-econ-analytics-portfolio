#!/usr/bin/env python
"""
pytorch_income_classifier.py

Simple PyTorch MLP classifier predicting high/low income from basic features.

Output:
    data/processed/income_classifier_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def main():
    print("Running pytorch_income_classifier.py ...")

    rng = np.random.default_rng(2025)
    n = 2000

    age = rng.integers(22, 65, size=n)
    edu = rng.choice([12, 14, 16, 18], size=n, p=[0.25, 0.3, 0.3, 0.15])
    exp = np.clip(age - rng.integers(18, 30, size=n), 0, None)

    # Latent income
    base = 15_000 + 800 * edu + 400 * exp + 200 * age
    noise = rng.normal(0, 10_000, size=n)
    income = base + noise

    high_income = (income > 80_000).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "edu_years": edu,
            "experience": exp,
            "income": income,
            "high_income": high_income,
        }
    )

    # Features and labels
    X = df[["age", "edu_years", "experience"]].to_numpy(dtype=np.float32)
    y = df["high_income"].to_numpy(dtype=np.int64)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train briefly
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {total_loss:.3f}")

    # Save dataset
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "income_classifier_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved income classifier dataset to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
