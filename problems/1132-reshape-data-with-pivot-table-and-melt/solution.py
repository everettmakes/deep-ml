import pandas as pd

def solution(df):
    wide = pd.pivot_table(
        df,
        index='date',
        columns='product',
        values='sales',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    long = wide.melt(
        id_vars='date',
        var_name='product',
        value_name='sales'
    )

    return long.sort_values(["date", "product"]).reset_index(drop=True)