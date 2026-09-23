import pandas as pd

def solution(df):
    return (
        df[df["status"] == "completed"]
        .groupby("region", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )