# Building Energy Efficiency Prediction

A machine learning project for predicting **heating load** and **cooling load** of buildings based on architectural design parameters, using the [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency).

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Interactive Dashboard](#interactive-dashboard)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

## 🎯 Project Overview

This project predicts the **heating load (Y1)** and **cooling load (Y2)** of residential buildings based on eight architectural design features. The goal is to assist architects and engineers in optimizing building designs for energy efficiency during the early design phase.

**Key highlights:**
- **Dataset**: 768 buildings with 8 input features and 2 target variables
- **Models tested**: Linear Regression, Random Forest, XGBoost, SVR (with hyperparameter tuning)
- **Best model**: XGBoost achieves R² > 0.998 for heating and R² > 0.992 for cooling
- **Deliverables**: Jupyter notebook, trained models, and an interactive Streamlit dashboard

---

## 📊 Dataset

**Source**: [UCI Machine Learning Repository - Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) (ID: 242)

| Property | Value |
|----------|-------|
| **Samples** | 768 |
| **Features** | 8 input variables |
| **Targets** | 2 (Heating Load, Cooling Load) |
| **Missing Values** | None |
| **Duplicate Rows** | None |

The dataset contains 768 buildings simulated using Ecotect. Each building varies across 8 design parameters with resulting heating and cooling loads calculated.

**File**: `ENB2012_data.csv` (local copy included)

---

## 🏗️ Features

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| **X1 - Relative_Compactness** | Ratio of surface area to volume | Continuous | 0.62 – 0.98 |
| **X2 - Surface_Area** | Total surface area (m²) | Continuous | 514.5 – 808.5 |
| **X3 - Wall_Area** | Total wall area (m²) | Continuous | 245.0 – 416.5 |
| **X4 - Roof_Area** | Total roof area (m²) | Continuous | 110.25 – 220.5 |
| **X5 - Overall_Height** | Building height (m) | Continuous | 3.5 – 7.0 |
| **X6 - Orientation** | Cardinal orientation (2=North, 3=East, 4=South, 5=West) | Categorical (int) | 2 – 5 |
| **X7 - Glazing_Area** | Window-to-wall ratio | Continuous | 0.0 – 0.4 |
| **X8 - Glazing_Area_Distribution** | Distribution of glazing (0=uniform, 1=North, 2=East, 3=South, 4=West) | Categorical (int) | 0 – 5 |

**Targets:**
- **Y1 - Heating_Load** (kWh/m²): 6.01 – 43.10
- **Y2 - Cooling_Load** (kWh/m²): 10.90 – 48.03

> **Note**: The geometric identity `Surface_Area = Wall_Area + 2 × Roof_Area` holds for all 768 rows, indicating redundancy among geometric features.

---

## 🔍 Exploratory Data Analysis

### Data Quality
- ✅ **768 rows × 10 columns** — no missing values
- ✅ **0 duplicate rows**
- ✅ All features populated across the full design space

### Key Correlations with Targets

| Feature | Heating Load | Cooling Load |
|---------|--------------|--------------|
| **Overall_Height** | **+0.889** | **+0.896** |
| **Roof_Area** | **−0.862** | **−0.863** |
| **Surface_Area** | −0.658 | −0.673 |
| Relative_Compactness | +0.622 | +0.634 |
| Wall_Area | +0.456 | +0.427 |
| Glazing_Area | +0.270 | +0.208 |
| Glazing_Area_Distribution | +0.087 | +0.051 |
| **Orientation** | **−0.003** | **+0.014** |

**Heating vs Cooling Load correlation**: **0.976** (very high)

### Visualizations Included
- Histograms of all variables
- Box plots for outlier detection
- Correlation heatmap
- Feature vs target scatter plots (16 plots)
- Pairwise relationships

---

## 🤖 Modeling Approach

### Pipeline
```
Raw Data → Train/Test Split (80/20, seed=42) → StandardScaler (fit on train only)
         → Model Training → Cross-Validation (5-fold) → Holdout Evaluation
```

### Models Evaluated

| Model | Type | Hyperparameters (Baseline) |
|-------|------|---------------------------|
| **Linear Regression** | Linear baseline | — |
| **Random Forest** | Ensemble (bagging) | n_estimators=100 |
| **XGBoost** | Gradient boosting | n_estimators=100 |
| **SVR** | Kernel method | RBF kernel, C=100 |

### Hyperparameter Tuning
- **Method**: RandomizedSearchCV (10 iterations, 3-fold CV)
- **Models tuned**: Random Forest, XGBoost, SVR
- **Metric**: Negative RMSE

#### Best Parameters Found

| Model | Target | Best Parameters |
|-------|--------|-----------------|
| Random Forest | Heating | `n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2` |
| XGBoost | Heating | `n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.85` |
| SVR | Heating | `C=1000, gamma='scale', epsilon=0.1` |
| Random Forest | Cooling | `n_estimators=100, max_depth=30, min_samples_split=10, min_samples_leaf=1` |
| XGBoost | Cooling | `n_estimators=300, max_depth=4, learning_rate=0.2, subsample=0.7` |
| SVR | Cooling | `C=1000, gamma='scale', epsilon=0.1` |

---

## 📈 Results

### Heating Load Prediction (Y1)

| Model | MAE | MSE | RMSE | R² | CV RMSE |
|-------|-----|-----|------|-----|---------|
| **XGBoost (tuned)** | **0.2780** | **0.1507** | **0.3882** | **0.9986** | 0.3688 |
| XGBoost (baseline) | 0.2636 | 0.1659 | 0.4074 | 0.9984 | 0.4005 |
| SVR (tuned) | 0.3451 | 0.2096 | 0.4578 | 0.9980 | 0.6248 |
| Random Forest (baseline) | 0.3540 | 0.2435 | 0.4934 | 0.9977 | 0.5153 |
| Random Forest (tuned) | 0.3640 | 0.2745 | 0.5240 | 0.9974 | 0.6119 |
| SVR (baseline) | 0.5614 | 0.5398 | 0.7347 | 0.9948 | 1.1637 |
| Linear Regression | 2.1820 | 9.1532 | 3.0254 | 0.9122 | 2.9292 |

### Cooling Load Prediction (Y2)

| Model | MAE | MSE | RMSE | R² | CV RMSE |
|-------|-----|-----|------|-----|---------|
| **XGBoost (baseline)** | **0.4486** | **0.7367** | **0.8583** | **0.9920** | **1.0879** |
| XGBoost (tuned) | 0.5653 | 0.9660 | 0.9828 | 0.9896 | 0.8872 |
| SVR (tuned) | 0.8542 | 2.1508 | 1.4665 | 0.9768 | 1.5390 |
| SVR (baseline) | 1.1369 | 3.1917 | 1.7865 | 0.9656 | 1.9475 |
| Random Forest (baseline) | 1.0720 | 2.9612 | 1.7208 | 0.9680 | 1.6888 |
| Random Forest (tuned) | 1.2283 | 3.6408 | 1.9081 | 0.9607 | 1.7700 |
| Linear Regression | 2.1953 | 9.8934 | 3.1454 | 0.8932 | 3.2292 |

> **Note**: For cooling load, the baseline XGBoost slightly outperforms the tuned version on the holdout set (likely due to small test set variance). Both are excellent.

### Summary

| Target | Best Model | RMSE | R² |
|--------|------------|------|-----|
| **Heating Load** | XGBoost (tuned) | **0.388** | **0.9986** |
| **Cooling Load** | XGBoost (baseline) | **0.858** | **0.9920** |

Both models achieve **near-perfect predictions** with R² > 0.99.

---

## 🎨 Feature Importance (Random Forest)

### Heating Load
1. **Relative_Compactness** — 39.1%
2. **Surface_Area** — 21.3%
3. **Wall_Area** — 14.3%
4. **Roof_Area** — 12.2%
5. **Overall_Height** — 8.9%
6. **Orientation** — 7.8%
7. **Glazing_Area** — 3.2%
8. **Glazing_Area_Distribution** — 1.7%

### Cooling Load
1. **Relative_Compactness** — 37.4%
2. **Roof_Area** — 18.9%
3. **Wall_Area** — 14.3%
4. **Surface_Area** — 12.2%
5. **Overall_Height** — 11.2%
6. **Glazing_Area** — 8.9%
7. **Orientation** — 9.8%
8. **Glazing_Area_Distribution** — 7.6%

**Key insight**: `Relative_Compactness` is the dominant feature for both targets. `Overall_Height` (highest correlation) ranks lower in tree-based importance due to correlation with other geometric features.

---

## 🌐 Interactive Dashboard

A **Streamlit-based dashboard** (`streamlit_app.py`) provides interactive exploration:

### Pages
1. **📊 Data Overview** — Dataset preview, statistics, missing/duplicate checks, download
2. **📈 EDA Visualizations** — Distributions, box plots, correlation matrix, scatter plots, pair plots, 3D scatter
3. **🔍 Feature Analysis** — Per-feature distributions, box/violin plots, target relationships, pair plots
4. **🤖 Model Performance** — Holdout metrics, CV scores, best model highlight
5. **📊 Model Comparison** — Interactive bar charts (R², RMSE, MAE), summary tables
6. **🎯 Feature Importance** — RF importance, permutation importance, correlation with targets

### Screenshots

*Dashboard screenshots can be added here after running the app*

---

## 💾 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to the project
cd energy_efficiency_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.5.0
seaborn>=0.12.0
streamlit>=1.20.0
plotly>=5.10.0
ucimlrepo>=0.0.5
jupyter>=1.0.0
notebook>=6.5.0
```

---

## 🚀 Usage

### 1. Run the Jupyter Notebook
```bash
jupyter notebook energy_efficiency.ipynb
```
Execute cells sequentially to reproduce the full analysis and modeling pipeline.

### 2. Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501` by default.

### 3. Quick Inference (from notebook)
```python
import joblib
import numpy as np

# Load trained model (after running notebook)
model = joblib.load('xgboost_heating_tuned.pkl')  # Example

# Predict for new building design
features = np.array([[0.75, 650, 300, 175, 5.0, 3, 0.2, 2]])  # 8 features
heating_load = model.predict(features)
```

---

## 📁 Project Structure

```
energy_efficiency_project/
│
├── ENB2012_data.csv          # Raw dataset (768 × 10)
├── energy_efficiency.ipynb   # Full analysis & modeling notebook
├── streamlit_app.py          # Interactive dashboard
├── model_comparison.csv      # Model metrics summary
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── __pycache__/              # Python bytecode (auto-generated)
│
└── (optional model artifacts after notebook run)
    ├── xgboost_heating_tuned.pkl
    ├── xgboost_cooling_tuned.pkl
    ├── scaler.pkl
    └── ...
```

---

## 💡 Key Findings

1. **XGBoost dominates** — Achieves R² > 0.99 for both targets with minimal tuning
2. **Geometry drives loads** — `Overall_Height` (positive) and `Roof_Area` (negative) are strongest linear correlates
3. **Orientation matters little** — Near-zero correlation (|r| < 0.02) with both targets
4. **Glazing distribution is weak** — Low correlation and low feature importance
5. **High target correlation** (0.976) suggests multi-output modeling could be effective
6. **Geometric redundancy** — `Surface_Area = Wall_Area + 2×Roof_Area` means 3 features encode 2 degrees of freedom
7. **Tree models capture non-linearity** — Scatter plots reveal banded structure from repeated design settings; linear models underfit

---

## 🔮 Future Work

- [ ] **Multi-output regression** — Leverage Y1/Y2 correlation (0.976) with `MultiOutputRegressor` or native multi-target models
- [ ] **Feature engineering** — Create interaction terms (e.g., `Glazing_Area × Orientation`), drop redundant geometric features
- [ ] **Model persistence** — Save best models with `joblib` for production inference
- [ ] **SHAP explanations** — Add model interpretability to dashboard
- [ ] **Deployment** — Containerize with Docker, deploy to cloud (Streamlit Cloud, Hugging Face Spaces, etc.)
- [ ] **Active learning** — Design optimal next simulations for expanding the dataset
- [ ] **Uncertainty quantification** — Prediction intervals via quantile regression or conformal prediction

---

## 📚 References

1. **UCI Dataset**: Tsanas, A. and Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools*. Energy and Buildings, 49, 560-567.
2. **Original Data**: Ecotect building performance simulation software
3. **Scikit-learn**: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825-2830.
4. **XGBoost**: Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.

---

## 👨‍💻 Author

**5th Semester Project**  
*Building Energy Efficiency Prediction using Machine Learning*

---

## 📄 License

This project is for educational purposes. The UCI Energy Efficiency dataset is publicly available under the [UCI terms of use](https://archive.ics.uci.edu/dataset/242/energy+efficiency).

---

*Generated as part of academic coursework. For questions or collaboration, please open an issue.*