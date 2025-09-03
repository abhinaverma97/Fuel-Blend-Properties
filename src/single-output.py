import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.model_selection import KFold
from scipy.stats import skew
from catboost import CatBoostRegressor

# Load train and test data
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
target_cols = [col for col in train.columns if col.startswith('BlendProperty')]

def add_engineered_features(df):
    for comp in range(1, 6):
        frac_col = f'Component{comp}_fraction'
        for prop in range(1, 11):
            prop_col = f'Component{comp}_Property{prop}'
            new_col = f'{frac_col}_x_{prop_col}'
            if frac_col in df.columns and prop_col in df.columns:
                df[new_col] = df[frac_col] * df[prop_col]
    for prop in range(1, 11):
        weighted_sum = 0
        for comp in range(1, 6):
            frac_col = f'Component{comp}_fraction'
            prop_col = f'Component{comp}_Property{prop}'
            if frac_col in df.columns and prop_col in df.columns:
                weighted_sum += df[frac_col] * df[prop_col]
        df[f'WeightedSum_Property{prop}'] = weighted_sum
    return df

train_fe = add_engineered_features(train.copy())
test_fe = add_engineered_features(test.copy())
all_features = [col for col in train_fe.columns if col not in target_cols]

# Per-target feature selection
N_FEATURES = 25
per_target_features = {}
for target in target_cols:
    X = train_fe[all_features]
    y = train_fe[target]
    lgb = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, force_col_wise=True)
    lgb.fit(X, y)
    importances = pd.Series(lgb.feature_importances_, index=all_features)
    top_features = importances.sort_values(ascending=False).head(N_FEATURES).index.tolist()
    per_target_features[target] = top_features

# Define base learners
base_learners = [
    ("xgb1", XGBRegressor(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42,tree_method='gpu_hist',verbosity=0)),
    ("lgbm2", LGBMRegressor(device='gpu', random_state=42, n_estimators=200, learning_rate=0.05, verbosity=-1, force_col_wise=True)),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("extratrees", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("ridge2", Ridge(alpha=0.1, random_state=42)),
    ("bayesianridge", BayesianRidge()),
    ("catboost", CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, l2_leaf_reg=3, random_state=42, verbose=False, task_type='GPU')),
]
meta_learner = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, tree_method='gpu_hist', verbosity=0)

# OOF stacking implementation with proper cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

test_preds = np.zeros((len(test_fe), len(target_cols)))
# Add OOF predictions array
oof_preds_all = np.zeros((len(train_fe), len(target_cols)))

for i, target in enumerate(target_cols):
    print(f"Processing target {target} ({i+1}/{len(target_cols)}) ...", flush=True)
    
    X_tr = train_fe[per_target_features[target]]
    y_tr = train_fe[target]
    X_test = test_fe[per_target_features[target]]
    
    # Ensure X_tr and y_tr are pandas DataFrame/Series for iloc
    if not isinstance(X_tr, pd.DataFrame):
        X_tr = pd.DataFrame(X_tr)
    if not isinstance(y_tr, pd.Series):
        y_tr = pd.Series(y_tr)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Target transformation
    target_skew = skew(y_tr)
    use_log = np.abs(target_skew) > 0.75
    shift = 0.0
    if use_log:
        min_y = y_tr.min()
        if min_y <= -1:
            shift = -min_y + 1.01
            y_tr_trans = np.log1p(y_tr + shift)
        else:
            y_tr_trans = np.log1p(y_tr)
    else:
        y_tr_trans = y_tr
    
    # Ensure y_tr_trans is a pandas Series for iloc
    if not isinstance(y_tr_trans, pd.Series):
        y_tr_trans = pd.Series(y_tr_trans)
    
    # Generate OOF predictions for base learners
    oof_preds = np.zeros((len(X_tr), len(base_learners)))
    test_preds_folds = np.zeros((len(X_test), len(base_learners), n_folds))
    # Add OOF meta-learner predictions for this target
    oof_meta_preds = np.zeros(len(X_tr))
    
    print(f"  Generating OOF predictions for target {target} ...", flush=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        X_train_fold = X_tr.iloc[train_idx]
        y_train_fold = y_tr_trans.iloc[train_idx]
        X_val_fold = X_tr.iloc[val_idx]
        
        # Train base learners on this fold
        for j, (name, model) in enumerate(base_learners):
            model_fold = type(model)(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
            
            # OOF predictions
            oof_preds[val_idx, j] = model_fold.predict(X_val_fold)
            
            # Test predictions for this fold
            test_preds_folds[:, j, fold] = model_fold.predict(X_test)
        
        # Meta-learner OOF prediction for this fold
        meta_learner_final = type(meta_learner)(**meta_learner.get_params())
        meta_features_val = np.column_stack([oof_preds[val_idx], X_val_fold])
        meta_learner_final.fit(np.column_stack([oof_preds[train_idx], X_train_fold]), y_tr_trans.iloc[train_idx])
        oof_meta_preds[val_idx] = meta_learner_final.predict(meta_features_val)
    
    # Inverse transform if log was used
    if use_log:
        if min_y <= -1:
            oof_meta_preds = np.expm1(oof_meta_preds) - shift
        else:
            oof_meta_preds = np.expm1(oof_meta_preds)
    
    # Post-processing
    min_y_val, max_y_val = y_tr.min(), y_tr.max()
    oof_meta_preds = np.clip(oof_meta_preds, min_y_val, max_y_val)
    oof_preds_all[:, i] = oof_meta_preds
    
    # Train meta-learner on OOF predictions with original features
    print(f"  Training meta-learner for target {target} ...", flush=True)
    meta_learner_final = type(meta_learner)(**meta_learner.get_params())
    # Include original features with OOF predictions for richer meta-features
    meta_features = np.column_stack([oof_preds, X_tr])
    meta_learner_final.fit(meta_features, y_tr_trans)
    
    # Average test predictions across folds and predict with meta-learner
    print(f"  Making final predictions for target {target} ...", flush=True)
    test_preds_avg = np.mean(test_preds_folds, axis=2)
    # Include original test features in meta-features for prediction
    test_meta_features = np.column_stack([test_preds_avg, X_test])
    y_pred = meta_learner_final.predict(test_meta_features)
    
    # Inverse transform if log was used
    if use_log:
        if min_y <= -1:
            y_pred = np.expm1(y_pred) - shift
        else:
            y_pred = np.expm1(y_pred)
    
    # Post-processing
    min_y_val, max_y_val = y_tr.min(), y_tr.max()
    y_pred = np.clip(y_pred, min_y_val, max_y_val)
    test_preds[:, i] = y_pred
    
    print(f"Completed target {target} ({i+1}/{len(target_cols)})", flush=True)

# Save submission
submission = pd.DataFrame(test_preds, columns=pd.Index(target_cols))
submission.insert(0, 'ID', np.arange(1, len(submission) + 1))
submission.to_csv('../outputs/single-output.csv', index=False)
print('Exp OOF stacking test predictions saved to ../outputs/single-output.csv')

# Save OOF predictions
submission_oof = pd.DataFrame(oof_preds_all, columns=pd.Index(target_cols))
submission_oof.insert(0, 'ID', np.arange(1, len(submission_oof) + 1))
submission_oof.to_csv('../outputs/oof-single-output.csv', index=False)
print('OOF predictions saved to ../outputs/oof-single-output.csv') 