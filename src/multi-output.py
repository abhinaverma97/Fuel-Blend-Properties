import os
import warnings
from sklearn.exceptions import ConvergenceWarning

os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from scipy.stats import skew
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump
import json

# Load train and test data
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
target_cols = [col for col in train.columns if col.startswith('BlendProperty')]

def add_engineered_features(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Dictionary to store all new features
    new_features = {}
    
    # Original interaction features
    for comp in range(1, 6):
        frac_col = f'Component{comp}_fraction'
        for prop in range(1, 11):
            prop_col = f'Component{comp}_Property{prop}'
            new_col = f'{frac_col}_x_{prop_col}'
            if frac_col in df.columns and prop_col in df.columns:
                new_features[new_col] = df[frac_col] * df[prop_col]
    
    # Weighted sum features (original)
    for prop in range(1, 11):
        weighted_sum = 0
        for comp in range(1, 6):
            frac_col = f'Component{comp}_fraction'
            prop_col = f'Component{comp}_Property{prop}'
            if frac_col in df.columns and prop_col in df.columns:
                weighted_sum += df[frac_col] * df[prop_col]
        new_features[f'WeightedSum_Property{prop}'] = weighted_sum
    
    # Component-level summary features
    for comp in range(1, 6):
        prop_cols = [f'Component{comp}_Property{i}' for i in range(1, 11)]
        if all(col in df.columns for col in prop_cols):
            new_features[f'Component{comp}_mean'] = df[prop_cols].mean(axis=1)
            new_features[f'Component{comp}_std'] = df[prop_cols].std(axis=1)
            new_features[f'Component{comp}_range'] = df[prop_cols].max(axis=1) - df[prop_cols].min(axis=1)
            new_features[f'Component{comp}_median'] = df[prop_cols].median(axis=1)
    
    # Property-level summary features across components
    for prop in range(1, 11):
        comp_cols = [f'Component{i}_Property{prop}' for i in range(1, 6)]
        if all(col in df.columns for col in comp_cols):
            new_features[f'Property{prop}_mean'] = df[comp_cols].mean(axis=1)
            new_features[f'Property{prop}_std'] = df[comp_cols].std(axis=1)
            new_features[f'Property{prop}_range'] = df[comp_cols].max(axis=1) - df[comp_cols].min(axis=1)
    
    # Component proportion ratios
    for i in range(1, 5):
        for j in range(i+1, 6):
            frac_i = f'Component{i}_fraction'
            frac_j = f'Component{j}_fraction'
            if frac_i in df.columns and frac_j in df.columns:
                new_features[f'Component{i}_to_Component{j}_ratio'] = (
                    df[frac_i] / (df[frac_j] + 1e-8)
                )
    
    # Cross-component property interactions
    for prop in range(1, 11):
        for i in range(1, 4):
            for j in range(i+1, 5):
                prop_i = f'Component{i}_Property{prop}'
                prop_j = f'Component{j}_Property{prop}'
                if prop_i in df.columns and prop_j in df.columns:
                    new_features[f'Property{prop}_Component{i}_x_Component{j}'] = df[prop_i] * df[prop_j]
    
    # Fraction-weighted property combinations
    for prop in range(1, 11):
        weighted_props = []
        for comp in range(1, 6):
            frac_col = f'Component{comp}_fraction'
            prop_col = f'Component{comp}_Property{prop}'
            if frac_col in df.columns and prop_col in df.columns:
                weighted_props.append(df[frac_col] * df[prop_col])
        
        if weighted_props:
            new_features[f'WeightedProduct_Property{prop}'] = np.prod(weighted_props, axis=0)
    
    # Global statistics across all components and properties
    all_prop_cols = [f'Component{i}_Property{j}' for i in range(1, 6) for j in range(1, 11)]
    if all(col in df.columns for col in all_prop_cols):
        new_features['Global_Property_mean'] = df[all_prop_cols].mean(axis=1)
        new_features['Global_Property_std'] = df[all_prop_cols].std(axis=1)
        new_features['Global_Property_max'] = df[all_prop_cols].max(axis=1)
        new_features['Global_Property_min'] = df[all_prop_cols].min(axis=1)
    
    # Add all new features at once using pd.concat
    if new_features:
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
    
    return df

def select_features_lgb_multioutput(X, y, n_features=75):
    """
    Select features using MultiOutputRegressor with LGBMRegressor.
    Aggregates feature importances across all targets and selects the top features.
    """
    lgb_multi = MultiOutputRegressor(LGBMRegressor(
        device='gpu', random_state=42, n_estimators=200, learning_rate=0.05, verbosity=-1, force_col_wise=True
    ))
    lgb_multi.fit(X, y)
    # Get feature importances for each target, skipping any None estimators
    importances = np.array([
        est.feature_importances_ for est in lgb_multi.estimators_ if est is not None
    ])
    # Aggregate (mean) across targets
    mean_importance = importances.mean(axis=0)
    feature_scores = pd.Series(mean_importance, index=X.columns)
    selected_features = feature_scores.nlargest(n_features).index.tolist()
    return selected_features

def remove_multicollinearity(X, threshold=0.95):
    """
    Remove highly correlated features to prevent ill-conditioned matrices.
    """
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation above threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    if to_drop:
        X_cleaned = X.drop(columns=to_drop)
        return X_cleaned, to_drop
    else:
        return X, []



train_fe = add_engineered_features(train.copy())
test_fe = add_engineered_features(test.copy())
all_features = [col for col in train_fe.columns if col not in target_cols]

# Select best features for multi-output modeling
selected_features = select_features_lgb_multioutput(train_fe[all_features], train_fe[target_cols], n_features=200)

# Remove multicollinearity
X_clean, removed_corr = remove_multicollinearity(train_fe[selected_features], threshold=0.95)
final_features = X_clean.columns.tolist()

print(f"Features: {len(all_features)} total, {len(selected_features)} selected, {len(removed_corr)} highly correlated removed, {len(final_features)} final")



# Define multi-output base learners with improved parameters
base_learners = [
    ("xgb_multi", MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, tree_method='hist', device='cuda', verbosity=0))),
    ("lgbm_multi", MultiOutputRegressor(LGBMRegressor(device='gpu', random_state=42, n_estimators=200, learning_rate=0.05, verbosity=-1, force_col_wise=True))),
    ("catboost_multi", MultiOutputRegressor(CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, l2_leaf_reg=3, random_state=42, verbose=False, task_type='GPU'))),
    ("ridge_multi", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42, solver='auto', tol=1e-4))),
    # ("lasso_multi", MultiOutputRegressor(Lasso(alpha=0.001, random_state=42, max_iter=2000, tol=1e-4, warm_start=True))),
    ("elasticnet_multi", MultiOutputRegressor(ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=2000, tol=1e-4, warm_start=True))),
    ("bayesianridge_multi", MultiOutputRegressor(BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6))),
]

# Multi-output meta learners
meta_learners = {
    "lgbm_multi": MultiOutputRegressor(LGBMRegressor(device='gpu', random_state=42, n_estimators=200, learning_rate=0.05, verbosity=-1, force_col_wise=True)),
}

# OOF stacking implementation with multi-output models
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

for meta_name, meta_learner in meta_learners.items():
    print(f"\n=== Meta-learner: {meta_name} ===")
    
    # Use final cleaned features for multi-output approach
    X_tr = train_fe[final_features]
    y_tr = train_fe[target_cols]
    X_test = test_fe[final_features]
    
    # Scale features to prevent convergence issues
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_test)
    X_tr_scaled = pd.DataFrame(X_tr_scaled, columns=X_tr.columns, index=X_tr.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Target transformation for each target
    y_tr_trans = y_tr.copy()
    shifts = {}
    use_log_flags = {}
    
    for target in target_cols:
        target_skew = skew(y_tr[target])
        use_log = np.abs(target_skew) > 0.75
        use_log_flags[target] = use_log
        shift = 0.0
        
        if use_log:
            min_y = y_tr[target].min()
            if min_y <= -1:
                shift = -min_y + 1.01
                y_tr_trans[target] = np.log1p(y_tr[target] + shift)
                shifts[target] = shift
            else:
                y_tr_trans[target] = np.log1p(y_tr[target])
                shifts[target] = 0.0
        else:
            shifts[target] = 0.0
    
    # Generate OOF predictions for base learners
    oof_preds = np.zeros((len(X_tr), len(base_learners), len(target_cols)))
    test_preds_folds = np.zeros((len(X_test), len(base_learners), len(target_cols), n_folds))
    
    # Arrays for storing metrics
    oof_predictions = np.zeros((len(X_tr), len(target_cols)))
    mae_scores = np.zeros(len(target_cols))
    mse_scores = np.zeros(len(target_cols))
    
    # Store trained base models for deployment
    trained_base_models = {}
    
    # Store trained base models for deployment
    trained_base_models = {}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_scaled)):
        # Ensure DataFrames with correct columns are used
        X_train_fold = X_tr_scaled.iloc[train_idx].copy()
        y_train_fold = y_tr_trans.iloc[train_idx].copy()
        X_val_fold = X_tr_scaled.iloc[val_idx].copy()
        
        # Train base learners on this fold
        for j, (name, model) in enumerate(base_learners):
            print(f"  {meta_name} - {name} - Fold {fold+1}/{n_folds}")
            # Create a fresh instance of the model for this fold
            if isinstance(model, MultiOutputRegressor):
                # For MultiOutputRegressor, create a new instance with the same estimator
                estimator = model.estimator
                model_fold = MultiOutputRegressor(type(estimator)(**estimator.get_params()))
            else:
                model_fold = type(model)(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
            
            # OOF predictions
            oof_preds[val_idx, j, :] = model_fold.predict(X_val_fold)
            
            # Test predictions for this fold (ensure DataFrame with correct columns)
            test_preds_folds[:, j, :, fold] = model_fold.predict(X_test_scaled.copy())
    
    # Train final base models on full dataset for deployment
    print(f"  Training final base models on full dataset...")
    for j, (name, model) in enumerate(base_learners):
        print(f"    Training final {name}")
        if isinstance(model, MultiOutputRegressor):
            estimator = model.estimator
            final_model = MultiOutputRegressor(type(estimator)(**estimator.get_params()))
        else:
            final_model = type(model)(**model.get_params())
        
        final_model.fit(X_tr_scaled, y_tr_trans)
        trained_base_models[name] = final_model
    
    # Train meta-learner on OOF predictions
    meta_learner = meta_learners[meta_name]
    if isinstance(meta_learner, MultiOutputRegressor):
        # For MultiOutputRegressor, create a new instance with the same estimator
        estimator = meta_learner.estimator
        meta_learner_final = MultiOutputRegressor(type(estimator)(**estimator.get_params()))
    else:
        meta_learner_final = type(meta_learner)(**meta_learner.get_params())
    
    # Reshape OOF predictions for meta-learner with proper feature names
    oof_preds_reshaped = oof_preds.reshape(len(X_tr_scaled), -1)
    
    # Create proper feature names for meta-features
    base_model_names = [name for name, _ in base_learners]
    meta_feature_names = []
    for model_name in base_model_names:
        for target_idx in range(len(target_cols)):
            meta_feature_names.append(f"{model_name}_target_{target_idx}")
    
    # Add original feature names
    meta_feature_names.extend(X_tr_scaled.columns.tolist())
    
    # Create DataFrame with proper feature names
    meta_features = np.column_stack([oof_preds_reshaped, X_tr_scaled])
    meta_features_df = pd.DataFrame(meta_features, columns=pd.Index(meta_feature_names), index=X_tr_scaled.index)
    
    meta_learner_final.fit(meta_features_df, y_tr_trans)
    
    # Calculate metrics for OOF predictions
    oof_meta_preds = meta_learner_final.predict(meta_features_df)
    
    # Inverse transform OOF predictions for metrics calculation
    for i, target in enumerate(target_cols):
        if use_log_flags[target]:
            if shifts[target] > 0:
                oof_meta_preds[:, i] = np.expm1(oof_meta_preds[:, i]) - shifts[target]
            else:
                oof_meta_preds[:, i] = np.expm1(oof_meta_preds[:, i])
        
        # Post-processing
        min_y_val, max_y_val = y_tr[target].min(), y_tr[target].max()
        oof_meta_preds[:, i] = np.clip(oof_meta_preds[:, i], min_y_val, max_y_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_tr[target], oof_meta_preds[:, i])
        mse = mean_squared_error(y_tr[target], oof_meta_preds[:, i])
        mae_scores[i] = mae
        mse_scores[i] = mse
        
        print(f"  {target} - MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    # Report overall metrics
    print("\n=== Overall Metrics ===")
    print(f"Mean MAE: {np.mean(mae_scores):.4f}")
    print(f"Mean MSE: {np.mean(mse_scores):.4f}")
    
    # Save metrics to CSV for further analysis
    metrics_df = pd.DataFrame({
        'Target': target_cols,
        'MAE': mae_scores,
        'MSE': mse_scores,
    })
    metrics_df.to_csv(f'../outputs/multi_output_metrics_{meta_name}.csv', index=False)
    print(f'  Saved: ../outputs/multi_output_metrics_{meta_name}.csv')
    
    # Save the trained meta-learner model
    dump(meta_learner_final, f'../outputs/meta_learner_{meta_name}.joblib')
    
    # Save the scaler
    dump(scaler, f'../outputs/scaler_{meta_name}.joblib')
    
    # Save trained base models
    dump(trained_base_models, f'../outputs/base_models_{meta_name}.joblib')
    
    # Save feature configuration and transformation parameters
    feature_config = {
        'final_features': final_features,
        'target_cols': target_cols,
        'shifts': shifts,
        'use_log_flags': {k: bool(v) for k, v in use_log_flags.items()},
        'meta_feature_names': meta_feature_names,
        'base_model_names': base_model_names,
        'target_min_max': {target: {'min': float(y_tr[target].min()), 'max': float(y_tr[target].max())} for target in target_cols}
    }
    
    with open(f'../outputs/feature_config_{meta_name}.json', 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    # Make final predictions
    test_preds_avg = np.mean(test_preds_folds, axis=3)
    test_preds_reshaped = test_preds_avg.reshape(len(X_test_scaled), -1)
    test_meta_features = np.column_stack([test_preds_reshaped, X_test_scaled])
    test_meta_features_df = pd.DataFrame(test_meta_features, columns=pd.Index(meta_feature_names), index=X_test_scaled.index)
    y_pred = meta_learner_final.predict(test_meta_features_df)
    
    # Inverse transform predictions
    for i, target in enumerate(target_cols):
        if use_log_flags[target]:
            if shifts[target] > 0:
                y_pred[:, i] = np.expm1(y_pred[:, i]) - shifts[target]
            else:
                y_pred[:, i] = np.expm1(y_pred[:, i])
        
        # Post-processing
        min_y_val, max_y_val = y_tr[target].min(), y_tr[target].max()
        y_pred[:, i] = np.clip(y_pred[:, i], min_y_val, max_y_val)
    
    # Save submission
    submission = pd.DataFrame(y_pred, columns=pd.Index(target_cols))
    submission.insert(0, 'ID', np.arange(1, len(submission) + 1))
    submission.to_csv(f'../outputs/multi_output_submission_{meta_name}.csv', index=False)
    print(f'  Saved: ../outputs/multi_output_submission_{meta_name}.csv')
    print(f'  Saved: ../outputs/meta_learner_{meta_name}.joblib')
    print(f'  Saved: ../outputs/scaler_{meta_name}.joblib')
    print(f'  Saved: ../outputs/base_models_{meta_name}.joblib')
    print(f'  Saved: ../outputs/feature_config_{meta_name}.json')

print("\n=== Multi-output stacking completed ===") 