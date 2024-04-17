import numpy as np
from hyperopt import fmin, tpe, hp, Trials
import numpy as np 


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from config import feature_path, score_path, lgb_params
from sklearn.model_selection import KFold, cross_val_score


import lightgbm as lgb

# Define search space for hyperparameters
space = {
    'num_leaves': hp.choice('num_leaves', range(20, 200)),
    'max_depth': hp.choice('max_depth', range(5, 15)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
    'bagging_freq': hp.choice('bagging_freq', range(1, 10)),
}




# Define objective function for optimization
def objective(params, X_train, y_train):
    # Convert integer parameters to int
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['bagging_freq'] = int(params['bagging_freq'])
    
    # Create LightGBM regressor with given hyperparameters
    model = lgb.LGBMRegressor(**params)
    
    # Perform cross-validation
    #kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Return negative mean squared error (to maximize)
    return -scores.mean()





    
if __name__ == '__main__':
    features = np.load(feature_path)
    scores = np.load(score_path)
    
    # Initialize trials object to track the progress
    trials = Trials()

    # Run hyperparameter optimization
    best = fmin(fn=lambda params: objective(params, features, scores), space=space, algo=tpe.suggest, trials=trials, max_evals=50)

    # Print best hyperparameters
    print("Best hyperparameters:", best)
    #print(features.shape, scores.shape)
    #train_lgbm_regressor(features, scores)
    