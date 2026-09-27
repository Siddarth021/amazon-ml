import pandas as pd
s1_df = pd.read_parquet('processed/train_s1.parquet').head(100)
c_df = pd.read_parquet('processed/train_candidate_pairs.parquet').head(100)
s1_cols = {'entity_id': 's1_id', 'clean_name': 'n1', 'clean_address': 'a1', 'numbers': 'nums1', 'pincodes': 'pins1', 'first_word': 'fw1', 'name_tokens': 'nt1', 'address_tokens': 'at1', 'country': 'country'}
merged = pd.merge(c_df, s1_df.rename(columns=s1_cols), left_on='source1_entity_id', right_on='s1_id', how='inner')
print(merged.columns)
