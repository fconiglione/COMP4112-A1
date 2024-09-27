import pandas as pd

data = pd.read_csv('email.tsv', sep='\t')
print(data.head())