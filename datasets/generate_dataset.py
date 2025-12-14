import numpy as np
import pandas as pd

def gerar_dataset(n):
    rng = np.random.default_rng()

    ids = np.arange(1, n + 1)

    salary_cents = rng.integers(500_00, 5000_00, size=n)
    bonus_cents = rng.integers(0, 1000_00, size=n)
    hours_per_week = rng.integers(20, 60, size=n)
    age = rng.integers(18, 70, size=n)

    df = pd.DataFrame({
        "id": ids,
        "salary_cents": salary_cents,
        "bonus_cents": bonus_cents,
        "hours_per_week": hours_per_week,
        "age": age,
    })

    return df

df = gerar_dataset(131072)
df.to_csv("dataset_131072.csv", index=False)