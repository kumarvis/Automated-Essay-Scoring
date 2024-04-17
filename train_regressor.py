import numpy as np 
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from config import feature_path, score_path, lgb_params
import pandas as pd

def split_train_test(features, scores):
    X_train, X_val, y_train, y_val = train_test_split(features, scores,
                                                    stratify=scores, 
                                                    test_size=0.2,
                                                    random_state=73)

    return X_train, X_val, y_train, y_val


def train_lgbm_regressor(features, scores):
    X_train, X_val, y_train, y_val = split_train_test(features, scores)
    # Convert datasets to LightGBM format
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Training the model
    num_round = 1000  # Number of boosting rounds
    print(f'Training Start')
    bst = lgb.train(lgb_params, train_data, num_round, valid_sets=[val_data])
    print(f'Training End')
    # Save model
    bst.save_model('lgbm_regressor_model.txt')
    
    # Predictions on validation set
    y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    y_pred_rnd = np.around(y_pred)
    # Calculate Root Mean Squared Error
    rmse = mean_squared_error(y_val, y_pred_rnd) ** 0.5
    print(f"Root Mean Squared Error on validation set: {rmse}") 

def predict_lgbm_regressor(features, scores):
    X_train, X_val, y_train, y_val = split_train_test(features, scores)
    # Convert datasets to LightGBM format
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.Booster(model_file='lgbm_regressor_model.txt')
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_rnd = np.around(y_pred)
    # Calculate Root Mean Squared Error
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    rmse_rnd = mean_squared_error(y_val, y_pred_rnd) ** 0.5
    print(f"Root Mean Squared Error on validation set: {rmse}") 
    print(f"Root Mean Squared Error Round on validation set: {rmse_rnd}") 
    
    
    df_result = pd.DataFrame({'y_val': y_val, 'y_pred': y_pred, 'y_pred_rnd': y_pred_rnd})
    df_result.to_csv('predict.csv', index=False)
    
    
if __name__ == '__main__':
    features = np.load(feature_path)
    scores = np.load(score_path)
    #print(features.shape, scores.shape)
    #train_lgbm_regressor(features, scores)
    predict_lgbm_regressor(features, scores)
    