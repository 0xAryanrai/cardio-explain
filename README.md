#  Heart Disease Prediction

A binary classification project that predicts the presence of heart disease using patient clinical data. The notebook walks through data cleaning, exploratory data analysis (EDA), feature encoding, and model comparison across three classifiers, with the best-performing model saved for reuse.

##  Dataset

The project uses the **UCI Heart Disease dataset** (`heart_disease_uci.csv`), which combines data from multiple clinical sites (e.g. Cleveland).

- **920 rows**, originally 16 columns
- Mix of numeric (age, blood pressure, cholesterol, etc.) and categorical (chest pain type, ECG results, etc.) features
- Target variable `num` (renamed to `Target`) originally ranges 0–4, indicating disease severity

## 🔧 Workflow

### 1. Data Preprocessing
- Dropped irrelevant columns: `id`, `dataset`
- Handled missing values:
  - Numeric columns (`trestbps`, `chol`, `thalch`, `oldpeak`, `ca`) → filled with **median**
  - Categorical columns (`fbs`, `restecg`, `exang`, `slope`, `thal`) → filled with **mode**
- Removed duplicate rows
- Converted the target into a **binary classification problem**: `0` = no disease, `1` = disease present (collapsing severity levels 1–4 into a single positive class)

### 2. Exploratory Data Analysis
- Correlation heatmap to identify feature relationships
- Identified `trestbps` (resting blood pressure) and `chol` (cholesterol) as the **least correlated** features with the target
- Class distribution visualized with a count plot and pie chart
- Pairplot to visualize feature interactions across the target classes

### 3. Feature Encoding
- Binary-encoded `sex` using `LabelEncoder`
- One-hot encoded remaining categorical features using `pd.get_dummies()`

### 4. Modeling
Data was split **80/20** into training and test sets (`random_state=42`). Three models were trained and evaluated on accuracy:

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.89% |
| Support Vector Machine (linear kernel) | 79.89% |
| **Random Forest** | **87.50%** 🏆 |

The **Random Forest Classifier** performed best and was selected as the final model.

### 5. Model Export
The trained Random Forest model and the list of feature columns it was trained on are saved using `joblib`:
- `best_heart_disease_model.pkl`
- `model_features.pkl`

This allows the model to be loaded later for inference without retraining.

## 🛠️ Tech Stack

- **Python**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, modeling, evaluation
- `joblib` — model persistence

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Run
1. Place `heart_disease_uci.csv` in the project directory
2. Open and run `Heart_disease_prediction.ipynb` in Jupyter Notebook / JupyterLab
3. The trained model will be exported as `best_heart_disease_model.pkl`

### Loading the saved model
```python
import joblib

model = joblib.load('best_heart_disease_model.pkl')
features = joblib.load('model_features.pkl')

# Ensure new input data has the same columns as `features` before predicting
prediction = model.predict(new_data[features])
```

##  Project Structure
```
.
├── Heart_disease_prediction.ipynb   # Main notebook
├── heart_disease_uci.csv            # Dataset (not included — add your own copy)
├── best_heart_disease_model.pkl     # Saved Random Forest model
├── model_features.pkl               # Feature columns used by the model
└── README.md
```

##  Future Improvements
- Hyperparameter tuning (e.g. `GridSearchCV`) for the Random Forest model
- Cross-validation instead of a single train/test split
- Model explainability with SHAP to interpret individual predictions
- Address missing-value-heavy columns (`slope`, `ca`, `thal`) with more robust imputation
- Deploy as a simple web app (e.g. Streamlit) for interactive predictions

## 📄 License
This project is open-source and available for educational use.
