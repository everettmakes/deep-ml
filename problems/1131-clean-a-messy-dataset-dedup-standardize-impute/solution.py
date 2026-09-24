import pandas as pd

def solution(df):
    df = df.copy()
    df['name'] = df['name'].str.strip().str.capitalize()
    df['date'] = pd.to_datetime(df['date'].str.strip()).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates(keep='first', ignore_index=True)
    df['value'] = df['value'].fillna(df['value'].mean())
    return df

