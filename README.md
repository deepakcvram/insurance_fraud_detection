# 🛡️ ClaimGuard AI: Vehicle Insurance Claim Fraud Detection

ClaimGuard AI is a premium machine learning dashboard designed to assess vehicle insurance claim fraud risk. Built specifically for insurance underwriters and risk auditors, this project demonstrates a complete production-grade machine learning workflow—handling severe class imbalance, deploying state-of-the-art gradient boosting classifiers, and implementing business-centric evaluation metrics like **Audit Efficiency (Lift)**.

### 🔗 Live Dashboard: https://vehicle-insurance-claim-fraud-detection.streamlit.app/

---

## 🚀 Key Features

*   **XGBoost Classifier**: Powered by an optimized Gradient Boosting model built for handling high-dimensional tabular datasets.
*   **Imbalance-Resilient Training**: Implements cost-sensitive learning (`scale_pos_weight = 15.71`) to train effectively on minority fraud cases (~6% prevalence) without oversampling noise.
*   **Operational Threshold Tuning**: A global risk threshold slider allowing underwriters to tune the decision boundary in real-time, shifting focus between catching all fraud (**Recall**) and avoiding false alarms (**Precision**).
*   **Business-Centric Evaluation**: Replaces misleading standard accuracy with **Audit Efficiency (Lift)**, demonstrating a **2.7x efficiency multiplier** over random audits at the optimal threshold.
*   **Interactive Streamlit UI**: Sleek dark-mode interface featuring dynamic metrics, interactive risk assessment forms, and real-time feature importances.

---

## 📊 Performance Metrics (Test Set Evaluation)

| Decision Threshold | Recall (Fraud Caught) | Precision (Flag Accuracy) | Audit Efficiency (Lift) | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **0.50 (Default)** | 2.7% | 41.7% | 7.0x | 93.8% |
| **0.15 (Balanced)** | **63.8%** | **16.1%** | **2.7x** | **77.9%** |
| **0.10 (High Sensitivity)**| 84.3% | 14.1% | 2.4x | 68.1% |

---

## 📁 Repository Structure

```
├── .gitignore             # Prevents pushing large datasets & cache files
├── README.md              # Project documentation
├── requirements.txt       # Python library dependencies
├── dataset.csv            # Vehicle insurance claims dataset (15,419 records)
├── model.ipynb            # Jupyter Notebook for baseline prototyping & EDA
├── train_and_save.py      # Production training script (splits, encodes, fits XGBoost)
├── app.py                 # Streamlit dashboard script
├── model.pkl              # Serialized XGBoost model
├── encoder.pkl            # Serialized One-Hot Encoder pipeline
└── metadata.pkl           # Helper mappings, column alignments, & test split arrays
```

---

## 🛠️ Installation & Setup (Local Run)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
Ensure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training pipeline to generate the serialized models and metadata:
```bash
python train_and_save.py
```

### 4. Run the Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Data Processing & Feature Engineering

1.  **Anomaly Resolution**: Identified and resolved an age data anomaly where a subset of minor policyholders were placeholder-imputed with `0` years old (mapped to `16`).
2.  **Ordinal Mapping**: Standardized size, period, and tier structures (e.g., `VehiclePrice`, `AgeOfVehicle`, `PastNumberOfClaims`) to continuous ranks.
3.  **Nominal Columns**: Encoded high-cardinality categorical data (`Make`, `PolicyType`, `DayOfWeekClaimed`) using `OneHotEncoder(drop='first', handle_unknown='ignore')` fitted strictly on the training set to prevent data leakage.
4.  **Dimensionality**: Preprocessing expands the 29 base features to **86 model features** to capture non-linear relationships.

---

## 🔑 Technologies Used

*   **Core**: Python, Pandas, NumPy
*   **Machine Learning**: Scikit-Learn, XGBoost
*   **Inference Serialization**: Pickle
*   **Application Deployment**: Streamlit, Streamlit Community Cloud
