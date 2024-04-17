from config import test_dataset_path
import pandas as pd

# df_train = pd.read_csv(train_dataset_path)
# df_train.to_parquet('dataset/train.parquet')

df_train = pd.read_csv(test_dataset_path)
df_train.to_parquet('dataset/test.parquet')

print('done')