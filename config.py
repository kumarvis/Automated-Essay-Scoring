train_dataset_path = 'dataset/train.parquet'
test_dataset_path = 'dataset/test.parquet'
feature_path = 'dataset/feature_1536.npy'
score_path = 'dataset/score.npy'

# Define parameters for LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',  # Root Mean Squared Error
    'num_leaves': 112,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 2,
    'max_depth': 9,
    'verbose': 0,
}
