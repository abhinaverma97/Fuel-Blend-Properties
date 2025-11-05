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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import json
import time
from datetime import datetime
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create research output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
research_dir = f'../outputs/research_paper_{timestamp}'
os.makedirs(research_dir, exist_ok=True)
os.makedirs(f'{research_dir}/models', exist_ok=True)
os.makedirs(f'{research_dir}/metrics', exist_ok=True)
os.makedirs(f'{research_dir}/predictions', exist_ok=True)
os.makedirs(f'{research_dir}/features', exist_ok=True)
os.makedirs(f'{research_dir}/base_learners', exist_ok=True)
os.makedirs(f'{research_dir}/analysis', exist_ok=True)
os.makedirs(f'{research_dir}/figures', exist_ok=True)

print(f"Research outputs will be saved to: {research_dir}")
print("="*80)

# Initialize timing and tracking
overall_start_time = time.time()
timing_log = {}

# Load train and test data
load_start = time.time()
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
target_cols = [col for col in train.columns if col.startswith('BlendProperty')]
timing_log['data_loading'] = time.time() - load_start

# Save dataset statistics
dataset_stats = {
    'train_samples': len(train),
    'test_samples': len(test),
    'n_targets': len(target_cols),
    'n_original_features': len([col for col in train.columns if col not in target_cols]),
    'target_names': target_cols,
    'train_shape': train.shape,
    'test_shape': test.shape
}

# Analyze target distributions
target_analysis = {}
for target in target_cols:
    target_analysis[target] = {
        'mean': float(train[target].mean()),
        'std': float(train[target].std()),
        'min': float(train[target].min()),
        'max': float(train[target].max()),
        'median': float(train[target].median()),
        'skewness': float(skew(train[target])),
        'q25': float(train[target].quantile(0.25)),
        'q75': float(train[target].quantile(0.75))
    }

# Save target correlation matrix
target_corr = train[target_cols].corr()
target_corr.to_csv(f'{research_dir}/analysis/target_correlation_matrix.csv')

# Plot 1: Target Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(target_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Blend Properties', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/target_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/target_correlation_heatmap.pdf', bbox_inches='tight')
plt.close()

# Plot 2: Target Distribution Analysis
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()
for i, target in enumerate(target_cols):
    axes[i].hist(train[target], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'{target}\nSkewness: {skew(train[target]):.3f}', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=9)
    axes[i].set_ylabel('Frequency', fontsize=9)
    axes[i].grid(alpha=0.3)
plt.suptitle('Distribution of Blend Properties', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/target_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/target_distributions.pdf', bbox_inches='tight')
plt.close()

# Plot 3: Target Statistics Box Plot
fig, ax = plt.subplots(figsize=(14, 6))
train[target_cols].boxplot(ax=ax, patch_artist=True)
ax.set_title('Box Plot of Blend Properties', fontsize=16, fontweight='bold')
ax.set_xlabel('Blend Property', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.grid(alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/target_boxplot.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/target_boxplot.pdf', bbox_inches='tight')
plt.close()

with open(f'{research_dir}/analysis/dataset_statistics.json', 'w') as f:
    json.dump(dataset_stats, f, indent=2)

with open(f'{research_dir}/analysis/target_analysis.json', 'w') as f:
    json.dump(target_analysis, f, indent=2)

print(f"Dataset loaded: {len(train)} train, {len(test)} test samples")
print(f"Targets: {len(target_cols)}")
print("="*80)

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

# Feature engineering
fe_start = time.time()
train_fe = add_engineered_features(train.copy())
test_fe = add_engineered_features(test.copy())
all_features = [col for col in train_fe.columns if col not in target_cols]
timing_log['feature_engineering'] = time.time() - fe_start

# Save feature engineering information
feature_info = {
    'original_features': len([col for col in train.columns if col not in target_cols]),
    'engineered_features': len(all_features) - len([col for col in train.columns if col not in target_cols]),
    'total_features': len(all_features),
    'feature_types': {
        'fraction_property_interactions': 50,  # 5 components × 10 properties
        'weighted_sums': 10  # 10 properties
    },
    'all_feature_names': all_features
}

with open(f'{research_dir}/features/feature_engineering_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"Feature engineering completed: {len(all_features)} total features")
print(f"  - Original: {feature_info['original_features']}")
print(f"  - Engineered: {feature_info['engineered_features']}")
print("="*80)

# Per-target feature selection
N_FEATURES = 25
per_target_features = {}
feature_importance_dict = {}

fs_start = time.time()
print("Performing per-target feature selection...")
for target in target_cols:
    X = train_fe[all_features]
    y = train_fe[target]
    lgb = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, force_col_wise=True)
    lgb.fit(X, y)
    importances = pd.Series(lgb.feature_importances_, index=all_features)
    top_features = importances.sort_values(ascending=False).head(N_FEATURES).index.tolist()
    per_target_features[target] = top_features
    feature_importance_dict[target] = importances.to_dict()
    print(f"  {target}: {N_FEATURES} features selected")

timing_log['feature_selection'] = time.time() - fs_start

# Save feature selection results
for target in target_cols:
    feat_df = pd.DataFrame({
        'feature': per_target_features[target],
        'importance': [feature_importance_dict[target][f] for f in per_target_features[target]]
    })
    feat_df.to_csv(f'{research_dir}/features/selected_features_{target}.csv', index=False)

# Create feature selection matrix (which features selected for which targets)
feature_selection_matrix = pd.DataFrame(0, index=all_features, columns=target_cols)
for target in target_cols:
    for feat in per_target_features[target]:
        feature_selection_matrix.loc[feat, target] = 1

feature_selection_matrix.to_csv(f'{research_dir}/features/feature_selection_matrix.csv')

# Analyze feature overlap
feature_overlap = {}
for target in target_cols:
    feature_overlap[target] = per_target_features[target]

with open(f'{research_dir}/features/per_target_features.json', 'w') as f:
    json.dump(feature_overlap, f, indent=2)

# Plot 4: Feature Selection Heatmap
plt.figure(figsize=(14, 20))
feature_usage = feature_selection_matrix.sum(axis=1).sort_values(ascending=False)
top_features = feature_usage.head(50).index
sns.heatmap(feature_selection_matrix.loc[top_features], cmap='YlGnBu', 
            cbar_kws={'label': 'Selected'}, linewidths=0.5)
plt.title('Top 50 Features: Selection Across Targets', fontsize=16, fontweight='bold')
plt.xlabel('Blend Property', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/feature_selection_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/feature_selection_heatmap.pdf', bbox_inches='tight')
plt.close()

# Plot 5: Feature Usage Frequency
plt.figure(figsize=(12, 6))
feature_usage_sorted = feature_usage.head(30)
plt.barh(range(len(feature_usage_sorted)), feature_usage_sorted.values, color='steelblue', edgecolor='black')
plt.yticks(range(len(feature_usage_sorted)), feature_usage_sorted.index, fontsize=9)
plt.xlabel('Number of Targets Using Feature', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 30 Most Frequently Selected Features', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/feature_usage_frequency.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/feature_usage_frequency.pdf', bbox_inches='tight')
plt.close()

print(f"Feature selection completed in {timing_log['feature_selection']:.2f}s")
print("="*80)

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

# Save model configurations
model_config = {
    'base_learners': {},
    'meta_learner': {},
    'n_folds': 5,
    'random_seed': 42,
    'n_features_selected': N_FEATURES
}

for name, model in base_learners:
    model_config['base_learners'][name] = str(model.get_params())

model_config['meta_learner'] = str(meta_learner.get_params())

with open(f'{research_dir}/models/model_configuration.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print(f"Base learners: {len(base_learners)}")
for name, _ in base_learners:
    print(f"  - {name}")
print(f"Meta learner: XGBRegressor")
print("="*80)

# OOF stacking implementation with proper cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

test_preds = np.zeros((len(test_fe), len(target_cols)))
# Add OOF predictions array
oof_preds_all = np.zeros((len(train_fe), len(target_cols)))

# Initialize comprehensive tracking dictionaries
base_learner_oof_preds = {name: np.zeros((len(train_fe), len(target_cols))) for name, _ in base_learners}
base_learner_test_preds = {name: np.zeros((len(test_fe), len(target_cols))) for name, _ in base_learners}
fold_metrics = {target: {f'fold_{i}': {} for i in range(n_folds)} for target in target_cols}
per_target_metrics = {target: {} for target in target_cols}
target_transformation_info = {}
base_learner_training_times = {name: [] for name, _ in base_learners}
meta_learner_training_times = []
trained_models_per_target = {}

print("Starting OOF stacking training...")
print("="*80)

for i, target in enumerate(target_cols):
    target_start_time = time.time()
    print(f"\nProcessing target {target} ({i+1}/{len(target_cols)}) ...")
    print("-"*80)
    
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
    
    # Save transformation info
    target_transformation_info[target] = {
        'skewness': float(target_skew),
        'use_log_transform': bool(use_log),
        'shift': float(shift),
        'min_value': float(y_tr.min()),
        'max_value': float(y_tr.max())
    }
    
    print(f"  Skewness: {target_skew:.4f}, Log transform: {use_log}")
    
    # Ensure y_tr_trans is a pandas Series for iloc
    if not isinstance(y_tr_trans, pd.Series):
        y_tr_trans = pd.Series(y_tr_trans)
    
    # Generate OOF predictions for base learners
    oof_preds = np.zeros((len(X_tr), len(base_learners)))
    test_preds_folds = np.zeros((len(X_test), len(base_learners), n_folds))
    # Add OOF meta-learner predictions for this target
    oof_meta_preds = np.zeros(len(X_tr))
    
    print(f"  Generating OOF predictions for target {target} ...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        fold_start = time.time()
        print(f"    Fold {fold+1}/{n_folds}:", end=" ")
        
        X_train_fold = X_tr.iloc[train_idx]
        y_train_fold = y_tr_trans.iloc[train_idx]
        X_val_fold = X_tr.iloc[val_idx]
        y_val_fold = y_tr_trans.iloc[val_idx]
        
        # Train base learners on this fold
        for j, (name, model) in enumerate(base_learners):
            base_start = time.time()
            model_fold = type(model)(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
            
            # OOF predictions
            oof_preds[val_idx, j] = model_fold.predict(X_val_fold)
            
            # Test predictions for this fold
            test_preds_folds[:, j, fold] = model_fold.predict(X_test)
            
            base_time = time.time() - base_start
            base_learner_training_times[name].append(base_time)
        
        # Meta-learner OOF prediction for this fold
        meta_start = time.time()
        meta_learner_final = type(meta_learner)(**meta_learner.get_params())
        meta_features_val = np.column_stack([oof_preds[val_idx], X_val_fold])
        meta_learner_final.fit(np.column_stack([oof_preds[train_idx], X_train_fold]), y_tr_trans.iloc[train_idx])
        oof_meta_preds[val_idx] = meta_learner_final.predict(meta_features_val)
        meta_time = time.time() - meta_start
        meta_learner_training_times.append(meta_time)
        
        # Calculate fold metrics (on transformed scale)
        fold_mae = mean_absolute_error(y_val_fold, oof_meta_preds[val_idx])
        fold_mse = mean_squared_error(y_val_fold, oof_meta_preds[val_idx])
        
        fold_metrics[target][f'fold_{fold}'] = {
            'mae_transformed': float(fold_mae),
            'mse_transformed': float(fold_mse),
            'training_time': float(time.time() - fold_start)
        }
        
        print(f"MAE: {fold_mae:.4f}, Time: {time.time() - fold_start:.2f}s")
    
    # Store base learner OOF predictions for this target
    for j, (name, model) in enumerate(base_learners):
        base_learner_oof_preds[name][:, i] = oof_preds[:, j]
    
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
    
    # Calculate final OOF metrics on original scale
    oof_mae = mean_absolute_error(y_tr, oof_meta_preds)
    oof_mse = mean_squared_error(y_tr, oof_meta_preds)
    oof_rmse = np.sqrt(oof_mse)
    oof_r2 = r2_score(y_tr, oof_meta_preds)
    
    # Calculate MAPE only if all values are positive
    try:
        if (y_tr > 0).all() and (oof_meta_preds > 0).all():
            oof_mape = mean_absolute_percentage_error(y_tr, oof_meta_preds)
        else:
            oof_mape = np.nan
    except:
        oof_mape = np.nan
    
    per_target_metrics[target] = {
        'oof_mae': float(oof_mae),
        'oof_mse': float(oof_mse),
        'oof_rmse': float(oof_rmse),
        'oof_r2': float(oof_r2),
        'oof_mape': float(oof_mape) if not np.isnan(oof_mape) else None,
        'training_time': float(time.time() - target_start_time)
    }
    
    print(f"  OOF Metrics - MAE: {oof_mae:.4f}, RMSE: {oof_rmse:.4f}, R²: {oof_r2:.4f}")
    
    # Train meta-learner on OOF predictions with original features
    print(f"  Training final meta-learner for target {target} ...")
    final_meta_start = time.time()
    meta_learner_final = type(meta_learner)(**meta_learner.get_params())
    # Include original features with OOF predictions for richer meta-features
    meta_features = np.column_stack([oof_preds, X_tr])
    meta_learner_final.fit(meta_features, y_tr_trans)
    
    # Save trained meta-learner
    joblib.dump(meta_learner_final, f'{research_dir}/models/meta_learner_{target}.pkl')
    trained_models_per_target[target] = {
        'meta_learner_path': f'meta_learner_{target}.pkl',
        'features_used': per_target_features[target]
    }
    
    # Average test predictions across folds and predict with meta-learner
    print(f"  Making final predictions for target {target} ...")
    test_preds_avg = np.mean(test_preds_folds, axis=2)
    
    # Store base learner average test predictions
    for j, (name, model) in enumerate(base_learners):
        base_learner_test_preds[name][:, i] = test_preds_avg[:, j]
    
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
    
    print(f"Completed target {target} ({i+1}/{len(target_cols)}) - Total time: {time.time() - target_start_time:.2f}s")
    print("="*80)

# Save submission
submission = pd.DataFrame(test_preds, columns=pd.Index(target_cols))
submission.insert(0, 'ID', np.arange(1, len(submission) + 1))
submission.to_csv('../outputs/single-output.csv', index=False)
submission.to_csv(f'{research_dir}/predictions/test_predictions.csv', index=False)
print('\nTest predictions saved to ../outputs/single-output.csv')

# Save OOF predictions
submission_oof = pd.DataFrame(oof_preds_all, columns=pd.Index(target_cols))
submission_oof.insert(0, 'ID', np.arange(1, len(submission_oof) + 1))
submission_oof.to_csv('../outputs/oof-single-output.csv', index=False)
submission_oof.to_csv(f'{research_dir}/predictions/oof_predictions.csv', index=False)
print('OOF predictions saved to ../outputs/oof-single-output.csv')

# Save base learner predictions
print("\nSaving base learner predictions...")
for name in base_learner_oof_preds.keys():
    # OOF predictions
    oof_df = pd.DataFrame(base_learner_oof_preds[name], columns=target_cols)
    oof_df.insert(0, 'ID', np.arange(1, len(oof_df) + 1))
    oof_df.to_csv(f'{research_dir}/base_learners/oof_predictions_{name}.csv', index=False)
    
    # Test predictions
    test_df = pd.DataFrame(base_learner_test_preds[name], columns=target_cols)
    test_df.insert(0, 'ID', np.arange(1, len(test_df) + 1))
    test_df.to_csv(f'{research_dir}/base_learners/test_predictions_{name}.csv', index=False)

print("Base learner predictions saved")

# Calculate and save comprehensive metrics
print("\nCalculating comprehensive metrics...")

# Overall metrics summary
overall_metrics = {
    'mean_mae': float(np.mean([per_target_metrics[t]['oof_mae'] for t in target_cols])),
    'mean_mse': float(np.mean([per_target_metrics[t]['oof_mse'] for t in target_cols])),
    'mean_rmse': float(np.mean([per_target_metrics[t]['oof_rmse'] for t in target_cols])),
    'mean_r2': float(np.mean([per_target_metrics[t]['oof_r2'] for t in target_cols])),
    'std_mae': float(np.std([per_target_metrics[t]['oof_mae'] for t in target_cols])),
    'std_mse': float(np.std([per_target_metrics[t]['oof_mse'] for t in target_cols])),
    'std_rmse': float(np.std([per_target_metrics[t]['oof_rmse'] for t in target_cols])),
    'std_r2': float(np.std([per_target_metrics[t]['oof_r2'] for t in target_cols]))
}

# Per-target metrics DataFrame
metrics_df = pd.DataFrame(per_target_metrics).T
metrics_df.to_csv(f'{research_dir}/metrics/per_target_metrics.csv')

# Save fold-wise metrics
with open(f'{research_dir}/metrics/fold_wise_metrics.json', 'w') as f:
    json.dump(fold_metrics, f, indent=2)

# Calculate base learner individual performance
print("Calculating base learner individual performance...")
base_learner_metrics = {}
for name in base_learner_oof_preds.keys():
    base_learner_metrics[name] = {}
    for i, target in enumerate(target_cols):
        y_true = train_fe[target]
        y_pred = base_learner_oof_preds[name][:, i]
        
        # Apply same inverse transformation as meta-learner
        if target_transformation_info[target]['use_log_transform']:
            shift = target_transformation_info[target]['shift']
            if shift > 0:
                y_pred = np.expm1(y_pred) - shift
            else:
                y_pred = np.expm1(y_pred)
        
        # Clip predictions
        y_pred = np.clip(y_pred, y_true.min(), y_true.max())
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        base_learner_metrics[name][target] = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }

# Create base learner comparison DataFrame
base_learner_comparison = []
for name in base_learner_metrics.keys():
    row = {'model': name}
    for target in target_cols:
        row[f'{target}_mae'] = base_learner_metrics[name][target]['mae']
    row['mean_mae'] = np.mean([base_learner_metrics[name][t]['mae'] for t in target_cols])
    base_learner_comparison.append(row)

base_comp_df = pd.DataFrame(base_learner_comparison)
base_comp_df.to_csv(f'{research_dir}/metrics/base_learner_comparison.csv', index=False)

# Calculate stacking improvement
stacking_improvement = {}
for target in target_cols:
    # Find best base learner for this target
    best_base_mae = min([base_learner_metrics[name][target]['mae'] for name in base_learner_metrics.keys()])
    stacking_mae = per_target_metrics[target]['oof_mae']
    improvement = ((best_base_mae - stacking_mae) / best_base_mae) * 100
    
    stacking_improvement[target] = {
        'best_base_learner_mae': float(best_base_mae),
        'stacking_mae': float(stacking_mae),
        'improvement_percentage': float(improvement)
    }

stacking_imp_df = pd.DataFrame(stacking_improvement).T
stacking_imp_df.to_csv(f'{research_dir}/metrics/stacking_improvement.csv')

# Plot 6: Base Learner Performance Comparison
plt.figure(figsize=(14, 8))
base_comp_df_plot = base_comp_df.set_index('model')
base_comp_df_plot['mean_mae'].sort_values().plot(kind='barh', color='coral', edgecolor='black')
plt.xlabel('Mean MAE Across All Targets', fontsize=12)
plt.ylabel('Base Learner', fontsize=12)
plt.title('Base Learner Performance Comparison', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/base_learner_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/base_learner_performance.pdf', bbox_inches='tight')
plt.close()

# Plot 7: Stacking Improvement per Target
plt.figure(figsize=(12, 6))
improvement_values = [stacking_improvement[t]['improvement_percentage'] for t in target_cols]
colors = ['green' if x > 0 else 'red' for x in improvement_values]
plt.bar(range(len(target_cols)), improvement_values, color=colors, edgecolor='black', alpha=0.7)
plt.xticks(range(len(target_cols)), target_cols, rotation=45, ha='right')
plt.xlabel('Blend Property', fontsize=12)
plt.ylabel('Improvement (%)', fontsize=12)
plt.title('Stacking Improvement Over Best Base Learner', fontsize=16, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/stacking_improvement.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/stacking_improvement.pdf', bbox_inches='tight')
plt.close()

# Plot 8: Performance Metrics per Target
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MAE
axes[0, 0].bar(range(len(target_cols)), [per_target_metrics[t]['oof_mae'] for t in target_cols], 
               color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xticks(range(len(target_cols)))
axes[0, 0].set_xticklabels(target_cols, rotation=45, ha='right')
axes[0, 0].set_ylabel('MAE', fontsize=11)
axes[0, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3, axis='y')

# RMSE
axes[0, 1].bar(range(len(target_cols)), [per_target_metrics[t]['oof_rmse'] for t in target_cols], 
               color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xticks(range(len(target_cols)))
axes[0, 1].set_xticklabels(target_cols, rotation=45, ha='right')
axes[0, 1].set_ylabel('RMSE', fontsize=11)
axes[0, 1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')

# R²
axes[1, 0].bar(range(len(target_cols)), [per_target_metrics[t]['oof_r2'] for t in target_cols], 
               color='seagreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xticks(range(len(target_cols)))
axes[1, 0].set_xticklabels(target_cols, rotation=45, ha='right')
axes[1, 0].set_ylabel('R² Score', fontsize=11)
axes[1, 0].set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# Training Time
axes[1, 1].bar(range(len(target_cols)), [per_target_metrics[t]['training_time'] for t in target_cols], 
               color='mediumpurple', edgecolor='black', alpha=0.7)
axes[1, 1].set_xticks(range(len(target_cols)))
axes[1, 1].set_xticklabels(target_cols, rotation=45, ha='right')
axes[1, 1].set_ylabel('Time (seconds)', fontsize=11)
axes[1, 1].set_title('Training Time per Target', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.suptitle('Performance Metrics Across Blend Properties', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/performance_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/performance_metrics.pdf', bbox_inches='tight')
plt.close()

# Calculate base learner correlation (diversity analysis)
base_learner_names = list(base_learner_oof_preds.keys())
correlation_matrices = {}
for i, target in enumerate(target_cols):
    preds_dict = {name: base_learner_oof_preds[name][:, i] for name in base_learner_names}
    corr_df = pd.DataFrame(preds_dict).corr()
    correlation_matrices[target] = corr_df
    corr_df.to_csv(f'{research_dir}/analysis/base_learner_correlation_{target}.csv')

# Plot 9: Base Learner Diversity (average correlation)
avg_corr_matrix = pd.DataFrame(0.0, index=base_learner_names, columns=base_learner_names)
for target in target_cols:
    avg_corr_matrix += correlation_matrices[target]
avg_corr_matrix /= len(target_cols)

plt.figure(figsize=(10, 8))
sns.heatmap(avg_corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0.5,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Average Base Learner Prediction Correlation\n(Diversity Analysis)', 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/base_learner_diversity.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/base_learner_diversity.pdf', bbox_inches='tight')
plt.close()

# Calculate residuals
print("Calculating residuals...")
residuals_df = pd.DataFrame()
for i, target in enumerate(target_cols):
    residuals_df[target] = train_fe[target] - oof_preds_all[:, i]

residuals_df.to_csv(f'{research_dir}/analysis/oof_residuals.csv', index=False)

# Residual statistics
residual_stats = {}
for target in target_cols:
    residual_stats[target] = {
        'mean': float(residuals_df[target].mean()),
        'std': float(residuals_df[target].std()),
        'min': float(residuals_df[target].min()),
        'max': float(residuals_df[target].max()),
        'q25': float(residuals_df[target].quantile(0.25)),
        'q50': float(residuals_df[target].quantile(0.50)),
        'q75': float(residuals_df[target].quantile(0.75))
    }

with open(f'{research_dir}/analysis/residual_statistics.json', 'w') as f:
    json.dump(residual_stats, f, indent=2)

# Plot 10: Residual Analysis
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()
for i, target in enumerate(target_cols):
    axes[i].hist(residuals_df[target], bins=30, edgecolor='black', alpha=0.7, color='indianred')
    axes[i].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[i].set_title(f'{target}\nMean: {residual_stats[target]["mean"]:.4f}', 
                     fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Residual', fontsize=9)
    axes[i].set_ylabel('Frequency', fontsize=9)
    axes[i].grid(alpha=0.3)
plt.suptitle('Residual Distributions (Actual - Predicted)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/residual_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/residual_distributions.pdf', bbox_inches='tight')
plt.close()

# Plot 11: Predicted vs Actual for All Targets
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()
for i, target in enumerate(target_cols):
    y_true = train_fe[target]
    y_pred = oof_preds_all[:, i]
    
    axes[i].scatter(y_true, y_pred, alpha=0.5, s=10, color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add R² to plot
    r2 = per_target_metrics[target]['oof_r2']
    mae = per_target_metrics[target]['oof_mae']
    axes[i].text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}', 
                transform=axes[i].transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[i].set_xlabel('Actual', fontsize=9)
    axes[i].set_ylabel('Predicted', fontsize=9)
    axes[i].set_title(f'{target}', fontsize=10, fontweight='bold')
    axes[i].grid(alpha=0.3)
    axes[i].legend(fontsize=7, loc='lower right')

plt.suptitle('Predicted vs Actual Values (Out-of-Fold)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/predicted_vs_actual.pdf', bbox_inches='tight')
plt.close()

# Plot 12: Overall Model Performance Summary
fig, ax = plt.subplots(figsize=(10, 6))
metrics_summary = pd.DataFrame({
    'MAE': [per_target_metrics[t]['oof_mae'] for t in target_cols],
    'RMSE': [per_target_metrics[t]['oof_rmse'] for t in target_cols],
    'R²': [per_target_metrics[t]['oof_r2'] for t in target_cols]
}, index=target_cols)

# Normalize metrics for comparison (R² is already 0-1, normalize MAE and RMSE)
metrics_normalized = metrics_summary.copy()
metrics_normalized['MAE'] = 1 - (metrics_normalized['MAE'] / metrics_normalized['MAE'].max())
metrics_normalized['RMSE'] = 1 - (metrics_normalized['RMSE'] / metrics_normalized['RMSE'].max())

metrics_normalized.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.7)
ax.set_ylabel('Normalized Score (higher is better)', fontsize=12)
ax.set_xlabel('Blend Property', fontsize=12)
ax.set_title('Normalized Performance Metrics Across Targets', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/normalized_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/normalized_performance.pdf', bbox_inches='tight')
plt.close()

# Plot 13: Training Time Analysis
plt.figure(figsize=(12, 6))
avg_times = {name: np.mean(times) for name, times in base_learner_training_times.items()}
sorted_times = dict(sorted(avg_times.items(), key=lambda x: x[1], reverse=True))

plt.barh(range(len(sorted_times)), list(sorted_times.values()), color='teal', edgecolor='black', alpha=0.7)
plt.yticks(range(len(sorted_times)), list(sorted_times.keys()))
plt.xlabel('Average Training Time per Fold (seconds)', fontsize=12)
plt.ylabel('Base Learner', fontsize=12)
plt.title('Base Learner Training Time Comparison', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{research_dir}/figures/training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{research_dir}/figures/training_time_comparison.pdf', bbox_inches='tight')
plt.close()

# Save timing information
timing_log['total_training'] = time.time() - overall_start_time
timing_log['average_base_learner_time'] = {
    name: float(np.mean(times)) for name, times in base_learner_training_times.items()
}
timing_log['average_meta_learner_time'] = float(np.mean(meta_learner_training_times))

with open(f'{research_dir}/metrics/timing_log.json', 'w') as f:
    json.dump(timing_log, f, indent=2)

# Save transformation information
with open(f'{research_dir}/analysis/target_transformations.json', 'w') as f:
    json.dump(target_transformation_info, f, indent=2)

# Save trained model information
with open(f'{research_dir}/models/trained_models_info.json', 'w') as f:
    json.dump(trained_models_per_target, f, indent=2)

# Create comprehensive summary report
print("\nGenerating comprehensive summary report...")
summary_report = {
    'experiment_info': {
        'timestamp': timestamp,
        'total_training_time': timing_log['total_training'],
        'n_folds': n_folds,
        'n_base_learners': len(base_learners),
        'n_features_selected': N_FEATURES
    },
    'dataset_info': dataset_stats,
    'overall_performance': overall_metrics,
    'best_target': {
        'by_mae': min(per_target_metrics.items(), key=lambda x: x[1]['oof_mae'])[0],
        'by_r2': max(per_target_metrics.items(), key=lambda x: x[1]['oof_r2'])[0]
    },
    'worst_target': {
        'by_mae': max(per_target_metrics.items(), key=lambda x: x[1]['oof_mae'])[0],
        'by_r2': min(per_target_metrics.items(), key=lambda x: x[1]['oof_r2'])[0]
    },
    'average_stacking_improvement': float(np.mean([stacking_improvement[t]['improvement_percentage'] for t in target_cols])),
    'feature_engineering': feature_info,
    'output_directory': research_dir
}

with open(f'{research_dir}/SUMMARY_REPORT.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

# Print final summary
print("\n" + "="*80)
print("TRAINING COMPLETED - RESEARCH DATA SUMMARY")
print("="*80)
print(f"\nOutput directory: {research_dir}")
print(f"\nTotal training time: {timing_log['total_training']:.2f}s ({timing_log['total_training']/60:.2f} minutes)")
print(f"\nOverall Performance:")
print(f"  Mean MAE:  {overall_metrics['mean_mae']:.4f} ± {overall_metrics['std_mae']:.4f}")
print(f"  Mean RMSE: {overall_metrics['mean_rmse']:.4f} ± {overall_metrics['std_rmse']:.4f}")
print(f"  Mean R²:   {overall_metrics['mean_r2']:.4f} ± {overall_metrics['std_r2']:.4f}")
print(f"\nAverage Stacking Improvement: {summary_report['average_stacking_improvement']:.2f}%")
print(f"\nBest performing target (MAE): {summary_report['best_target']['by_mae']}")
print(f"Worst performing target (MAE): {summary_report['worst_target']['by_mae']}")

print("\n" + "="*80)
print("SAVED FILES:")
print("="*80)
print(f"\n📁 {research_dir}/")
print(f"  ├── predictions/")
print(f"  │   ├── test_predictions.csv")
print(f"  │   └── oof_predictions.csv")
print(f"  ├── metrics/")
print(f"  │   ├── per_target_metrics.csv")
print(f"  │   ├── fold_wise_metrics.json")
print(f"  │   ├── base_learner_comparison.csv")
print(f"  │   ├── stacking_improvement.csv")
print(f"  │   └── timing_log.json")
print(f"  ├── features/")
print(f"  │   ├── selected_features_[target].csv (x10)")
print(f"  │   ├── feature_selection_matrix.csv")
print(f"  │   ├── per_target_features.json")
print(f"  │   └── feature_engineering_info.json")
print(f"  ├── models/")
print(f"  │   ├── meta_learner_[target].pkl (x10)")
print(f"  │   ├── model_configuration.json")
print(f"  │   └── trained_models_info.json")
print(f"  ├── base_learners/")
print(f"  │   ├── oof_predictions_[model].csv (x7)")
print(f"  │   └── test_predictions_[model].csv (x7)")
print(f"  ├── analysis/")
print(f"  │   ├── dataset_statistics.json")
print(f"  │   ├── target_analysis.json")
print(f"  │   ├── target_correlation_matrix.csv")
print(f"  │   ├── base_learner_correlation_[target].csv (x10)")
print(f"  │   ├── oof_residuals.csv")
print(f"  │   ├── residual_statistics.json")
print(f"  │   └── target_transformations.json")
print(f"  ├── figures/")
print(f"  │   ├── target_correlation_heatmap.png/pdf")
print(f"  │   ├── target_distributions.png/pdf")
print(f"  │   ├── target_boxplot.png/pdf")
print(f"  │   ├── feature_selection_heatmap.png/pdf")
print(f"  │   ├── feature_usage_frequency.png/pdf")
print(f"  │   ├── base_learner_performance.png/pdf")
print(f"  │   ├── stacking_improvement.png/pdf")
print(f"  │   ├── performance_metrics.png/pdf")
print(f"  │   ├── base_learner_diversity.png/pdf")
print(f"  │   ├── residual_distributions.png/pdf")
print(f"  │   ├── predicted_vs_actual.png/pdf")
print(f"  │   ├── normalized_performance.png/pdf")
print(f"  │   └── training_time_comparison.png/pdf")
print(f"  └── SUMMARY_REPORT.json")
print("\n" + "="*80)
print("All research data has been saved successfully!")
print("You can now use this data for your research paper analysis and visualizations.")
print("="*80) 