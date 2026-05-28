import pandas as pd

df = pd.read_parquet('data/processed/darpa_tc_e3/theia/labeled/labeled_shard0.parquet')
print("Events Schema:")
print(df.dtypes)
print(df.head(2))

try:
    subjects = pd.read_parquet('data/processed/darpa_tc_e3/theia/subjects.parquet')
    print("\nSubjects Schema:")
    print(subjects.dtypes)
    print(subjects.head(2))
except Exception as e:
    print(e)
