# Fuel Blend Properties — Shell Hackathon 🏆

**Top 18 / ~7,000 participants** in Shell AI Hackathon 2025

Full-stack machine learning project for predicting fuel blend properties (BlendProperty1-10) using ensemble stacking methods and deployed via a modern web application. This repository contains training pipelines, model artifacts, and a production-ready web interface for real-time predictions.

## Project Highlights

- **Hybrid Ensemble Architecture**: Combined multi-output and per-target stacking to exploit correlations while capturing per-target nuances
- **Per-Target Feature Selection**: Optimized feature sets for each property, achieving 5% MAPE reduction vs baseline
- **Advanced Stacking Pipeline**: 7 base learners (XGBoost, LightGBM, CatBoost, Random Forest, ExtraTrees, Ridge, Bayesian Ridge) + XGBoost meta-learner
- **Production Web App**: Next.js 15 + Flask API with glass-morphism dark UI for real-time predictions
- **2% MAPE Improvement**: Through hybrid ensemble combining multi-output and per-target approaches

## Repository Structure

```
Shell/
├── dataset/              # Training and test data (train.csv, test.csv)
├── src/                  # ML training scripts
│   ├── single-output.py  # Per-target OOF stacking pipeline (main model)
│   ├── multi-output.py   # Multi-output modeling utilities
│   └── paper artifacts.py
├── models/               # Trained model artifacts
│   ├── meta_model_BlendProperty*.pkl  # 10 trained meta-learners
│   └── model_metadata.json            # Feature lists & preprocessing params
├── blend-predictor/      # Web application
│   ├── app/             # Next.js 15 frontend (TypeScript, Tailwind)
│   └── flask-api/       # Flask REST API backend
├── outputs/             # Prediction results & OOF files
├── reports/             # EDA and analysis notes
├── workflow.png         # System architecture diagram
└── requirements.txt     # Python dependencies
```

## Quick Start

### 1. Train Models (Optional - pre-trained models included)

```powershell
# Setup Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train per-target stacking models
python .\src\single-output.py
```

**Output**: 10 model files in `models/` + predictions in `outputs/`

### 2. Run Web Application

#### Start Flask API (Terminal 1)
```powershell
cd blend-predictor\flask-api
pip install Flask flask-cors numpy pandas scikit-learn xgboost
python app.py
```
API runs at `http://localhost:5000`

#### Start Next.js Frontend (Terminal 2)
```powershell
cd blend-predictor
npm install
npm run dev
```
Web app runs at `http://localhost:3000`

## Web Application Features

- **Responsive Glass-Morphism UI**: Dark mode with cyan/purple gradients
- **Real-time Predictions**: Submit component data, get instant blend property predictions
- **Random Data Generator**: Test with one click
- **Input Validation**: Ensures fractions sum to 1.0
- **60+ Input Fields**: 5 components × (1 fraction + 10 properties)
- **10 Prediction Outputs**: All blend properties displayed with progress bars

## ML Pipeline Architecture

### Feature Engineering
- **Interaction Features**: `Component{i}_fraction × Component{i}_Property{j}` (50 features)
- **Weighted Sums**: `Σ(fraction × property)` per property (10 features)
- **Total Features**: 60 base + 60 engineered = 120 features

### Per-Target Feature Selection
- LightGBM importance ranking
- Top 25 features selected per target
- Reduces noise and improves generalization

### Stacking Ensemble
1. **Base Learners** (7 models):
   - XGBoost, LightGBM, CatBoost
   - Random Forest, ExtraTrees
   - Ridge, Bayesian Ridge

2. **5-Fold Cross-Validation**:
   - Out-of-fold predictions for meta-features
   - Prevents data leakage

3. **Meta-Learner**:
   - XGBoost trained on base predictions + original features
   - Log transformation for skewed targets (|skew| > 0.75)
   - Min-max clipping to training range

## Results

- **Competition Rank**: Top 18 / ~7,000 participants
- **MAPE Reduction**: 2% improvement via hybrid ensemble
- **Feature Selection Gain**: 5% MAPE reduction vs baseline
- **API Response Time**: <200ms per prediction

## Tech Stack

### Machine Learning
- **Python 3.12**: Core language
- **scikit-learn**: Preprocessing, base models
- **XGBoost, LightGBM, CatBoost**: Gradient boosting
- **NumPy, Pandas**: Data manipulation

### Web Application
- **Frontend**: Next.js 15, TypeScript, Tailwind CSS, React Hooks
- **Backend**: Flask, Flask-CORS
- **Deployment**: RESTful API architecture
- **UI/UX**: Glass morphism, dark mode, responsive design

## Key Files

- `src/single-output.py` — Main training pipeline with OOF stacking
- `models/meta_model_*.pkl` — 10 trained meta-learner models
- `models/model_metadata.json` — Feature lists and preprocessing parameters
- `blend-predictor/app/page.tsx` — Next.js frontend component
- `blend-predictor/flask-api/app.py` — Flask prediction API

## GPU Acceleration

Scripts use GPU by default for XGBoost, LightGBM, and CatBoost. To run on CPU:

**XGBoost**: Change `tree_method='gpu_hist'` → `tree_method='hist'`  
**LightGBM**: Remove `device='gpu'`  
**CatBoost**: Remove `task_type='GPU'`

## Model Performance Notes

- **Target Transformation**: Automatic log1p for skewed targets
- **Negative Handling**: Shift adjustment for negative values before log
- **Reproducibility**: Random seeds set (42) for all models
- **Validation**: 5-fold stratified cross-validation

## 📄 License & Acknowledgements

This repository was created for the Shell AI Hackathon 2025. Feel free to use for learning and experimentation. If publishing results, please acknowledge the original work.

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- Performance improvements
- New features
- Documentation updates


