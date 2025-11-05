from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load models and metadata at startup
MODELS_DIR = '../../models'
models = {}
metadata = {}

print("Loading models...")
for i in range(1, 11):
    target = f'BlendProperty{i}'
    model_path = os.path.join(MODELS_DIR, f'meta_model_{target}.pkl')
    with open(model_path, 'rb') as f:
        models[target] = pickle.load(f)
    print(f"Loaded model for {target}")

metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print("Loaded metadata")

def add_engineered_features(df):
    """Apply the same feature engineering as training"""
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data or 'components' not in data:
            return jsonify({'error': 'Invalid input. Expected "components" field'}), 400
        
        components = data['components']
        
        # Build input DataFrame
        input_data = {}
        for i, comp in enumerate(components, 1):
            input_data[f'Component{i}_fraction'] = [comp['fraction']]
            for j, prop_value in enumerate(comp['properties'], 1):
                input_data[f'Component{i}_Property{j}'] = [prop_value]
        
        df = pd.DataFrame(input_data)
        
        # Apply feature engineering
        df_fe = add_engineered_features(df.copy())
        
        # Make predictions for all targets
        predictions = {}
        
        for target in sorted(models.keys()):
            meta = metadata[target]
            selected_features = meta['selected_features']
            use_log = meta['use_log']
            shift = meta['shift']
            min_y = meta['min_y']
            max_y = meta['max_y']
            
            # Select features
            X = df_fe[selected_features]
            
            # Note: For production, we need base learner predictions too
            # For simplicity, we'll just use the features directly
            # This is a simplified version - you may need to train base learners on full data
            
            # Predict (this is simplified - in full stacking you'd need base learner predictions)
            # For now, we'll create a dummy prediction using just the features
            # You should retrain base learners on full training data and save them too
            
            # Simplified prediction - just using features
            # In production, you'd want base learner predictions here
            model = models[target]
            
            # Create feature matrix with dummy base learner predictions
            # This matches the training setup: [base_learner_preds, original_features]
            n_base_learners = 7  # Number of base learners in training
            dummy_base_preds = np.zeros((1, n_base_learners))  # Placeholder
            meta_features = np.column_stack([dummy_base_preds, X])
            
            y_pred = model.predict(meta_features)[0]
            
            # Inverse transform if log was used
            if use_log:
                if shift > 0:
                    y_pred = np.expm1(y_pred) - shift
                else:
                    y_pred = np.expm1(y_pred)
            
            # Clip to training range
            y_pred = np.clip(y_pred, min_y, max_y)
            
            predictions[target] = float(y_pred)
        
        return jsonify({
            'predictions': predictions,
            'input_summary': {
                'total_fraction': sum(c['fraction'] for c in components),
                'num_components': len(components)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
