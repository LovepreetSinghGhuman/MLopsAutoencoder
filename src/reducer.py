import pandas as pd

df = pd.read_csv('data\processed\cleaned_test.csv')

reduced_df = df.head(1000)  # Reduce to 1000 rows for testing

reduced_df.to_csv('data/processed/reduced_test.csv', index=False)