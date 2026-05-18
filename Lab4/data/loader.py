from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "wine+quality"


def load_data():
    red_path = DATA_DIR / "winequality-red.csv"
    white_path = DATA_DIR / "winequality-white.csv"

    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    red["type"] = 0
    white["type"] = 1

    df = pd.concat([red, white], ignore_index=True)

    X = df.drop("quality", axis=1).values
    y = df["quality"].values

    return X, y