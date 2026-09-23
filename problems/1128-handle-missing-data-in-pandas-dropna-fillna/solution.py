import pandas as pd

def solution(df):
    col_missing = df.isnull().mean()
    df = df.loc[:, col_missing <= 0.5]

    row_missing = df.isnull().mean(axis=1)
    df = df.loc[row_missing <= 0.5].copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0])

    return df.reset_index(drop=True)