import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Set page config
st.set_page_config(
    page_title="Insurance Claim Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .title-text {
        background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        color: #ffffff;
    }
    
    .card {
        background-color: #151922;
        border: 1px solid #242b3b;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
        margin-top: 20px;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load helper function
@st.cache_resource
def load_assets():
    if not (os.path.exists("model.pkl") and os.path.exists("encoder.pkl") and os.path.exists("metadata.pkl")):
        return None, None, None
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, encoder, metadata

model, encoder, metadata = load_assets()

# Sidebar Risk Control
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #818cf8; margin-bottom: 0px;'>🛡️ ClaimGuard AI</h2>
        <p style='color: #64748b; font-size: 0.9rem;'>Risk Optimization Console</p>
    </div>
    <hr style='border: 1px solid #242b3b;' />
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🎛️ Risk Threshold Tuning")
st.sidebar.markdown("Adjust the probability threshold to balance the model's **Precision** and **Recall**:")
threshold = st.sidebar.slider(
    "Decision Threshold", 
    min_value=0.01, 
    max_value=1.00, 
    value=0.15, 
    step=0.01,
    help="Lowering the threshold catches more fraud (higher Recall) but increases false positives. Raising it reduces false positives (higher Precision) but misses fraud."
)

# Main app title
st.markdown("<div class='title-text'>Vehicle Insurance Claim Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>A premium machine learning dashboard for assessing risk and detecting suspicious insurance claims.</div>", unsafe_allow_html=True)

if model is None:
    st.error("❌ Model assets could not be found! Please run the training script (`train_and_save.py`) first to generate the pickle files.")
    st.stop()

# Set up tabs
tab1, tab2 = st.tabs(["🔍 Assess Claim Risk", "📊 Model Insights & Analytics"])

with tab1:
    st.markdown("### 📋 Enter Claim and Policy Details")
    st.markdown("Fill out the fields below and click **Assess Claim Risk** to check for potential fraud indicators.")
    
    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Policyholder Details")
            age = st.slider("Age of Driver", min_value=16, max_value=100, value=35)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widow"])
            age_holder = st.selectbox("Age of Policy Holder Group", [
                "31 to 35", "36 to 40", "41 to 50", "51 to 65", "26 to 30", "over 65", "16 to 17", "21 to 25", "18 to 20"
            ])
            rating = st.selectbox("Driver Rating", [1, 2, 3, 4])
            past_claims = st.selectbox("Past Number of Claims", ["none", "1", "2 to 4", "more than 4"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📅 Time of Accident")
            month = st.selectbox("Month of Accident", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            week = st.selectbox("Week of Month (Accident)", [1, 2, 3, 4, 5])
            day_of_week = st.selectbox("Day of Week (Accident)", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        with col2:
            st.markdown("#### 🚗 Vehicle Details")
            make = st.selectbox("Vehicle Make", [
                "Pontiac", "Toyota", "Honda", "Mazda", "Chevrolet", "Accura", "Ford", "VW", 
                "Dodge", "Saab", "Mercury", "Saturn", "Nisson", "BMW", "Jaguar", "Porche", 
                "Mecedes", "Ferrari", "Lexus"
            ])
            vehicle_cat = st.selectbox("Vehicle Category", ["Sedan", "Sport", "Utility"])
            age_vehicle = st.selectbox("Age of Vehicle", ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"])
            price = st.selectbox("Vehicle Price Range", ["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000"])
            num_cars = st.selectbox("Number of Cars in Policy", ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📅 Time of Claim")
            month_claimed = st.selectbox("Month Claimed", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            week_claimed = st.selectbox("Week of Month (Claimed)", [1, 2, 3, 4, 5])
            day_claimed = st.selectbox("Day of Week (Claimed)", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        with col3:
            st.markdown("#### 📄 Policy & Claim Details")
            policy_type = st.selectbox("Policy Type", [
                "Sedan - Collision", "Sedan - Liability", "Sedan - All Perils", "Sport - Collision", 
                "Utility - All Perils", "Utility - Collision", "Sport - All Perils", "Utility - Liability", "Sport - Liability"
            ])
            base_policy = st.selectbox("Base Policy", ["Collision", "Liability", "All Perils"])
            deductible = st.selectbox("Deductible ($)", [300, 400, 500, 700])
            days_policy_acc = st.selectbox("Days: Policy to Accident", ["more than 30", "15 to 30", "none", "1 to 7", "8 to 15"])
            days_policy_claim = st.selectbox("Days: Policy to Claim", ["more than 30", "15 to 30", "8 to 15", "none"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### ⚠️ Report & Witness Info")
            fault = st.radio("Fault", ["Policy Holder", "Third Party"], horizontal=True)
            police_reported = st.radio("Police Report Filed?", ["Yes", "No"], horizontal=True)
            witness = st.radio("Witness Present?", ["Yes", "No"], horizontal=True)
            agent = st.radio("Agent Type", ["External", "Internal"], horizontal=True)
            supplements = st.selectbox("Number of Supplements", ["none", "1 to 2", "3 to 5", "more than 5"])
            address_change = st.selectbox("Address Change vs Claim", ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"])
            area = st.radio("Accident Area", ["Urban", "Rural"], horizontal=True)

        submit = st.form_submit_button("Assess Claim Risk")

    if submit:
        # Preprocess input
        input_data = {
            'Age': age,
            'Sex': sex,
            'AccidentArea': area,
            'Fault': fault,
            'PoliceReportFiled': police_reported,
            'WitnessPresent': witness,
            'AgentType': agent,
            'VehiclePrice': price,
            'AgeOfVehicle': age_vehicle,
            'AgeOfPolicyHolder': age_holder,
            'Days_Policy_Accident': days_policy_acc,
            'Days_Policy_Claim': days_policy_claim,
            'PastNumberOfClaims': past_claims,
            'NumberOfSuppliments': supplements,
            'AddressChange_Claim': address_change,
            'NumberOfCars': num_cars,
            'Deductible': deductible,
            'DriverRating': rating,
            'WeekOfMonth': week,
            'WeekOfMonthClaimed': week_claimed,
            # Nominal ones will go into One-Hot Encoder
            'Make': make,
            'PolicyType': policy_type,
            'MaritalStatus': marital,
            'VehicleCategory': vehicle_cat,
            'BasePolicy': base_policy,
            'Month': month,
            'MonthClaimed': month_claimed,
            'DayOfWeek': day_of_week,
            'DayOfWeekClaimed': day_claimed
        }
        
        # 1. Apply Binary Mappings
        binary_mappings = metadata['binary_cols']
        for col, mapping in binary_mappings.items():
            input_data[col] = mapping[input_data[col]]
            
        # 2. Apply Ordinal Mappings
        ordinal_mappings = metadata['ordinal_mappings']
        for col, mapping in ordinal_mappings.items():
            input_data[col] = mapping[input_data[col]]
            
        # Convert to single-row dataframe for processing
        input_df = pd.DataFrame([input_data])
        
        # 3. Apply One-Hot Encoder on Nominal Columns
        nominal_cols = metadata['nominal_cols']
        encoded_cats = encoder.transform(input_df[nominal_cols])
        encoded_cols = encoder.get_feature_names_out(nominal_cols)
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols, index=input_df.index)
        
        # Drop original nominal columns and join encoded ones
        input_preprocessed = input_df.drop(columns=nominal_cols).join(encoded_df)
        
        # Align with original feature columns (ensure correct column order/presence)
        final_cols = metadata['feature_columns']
        input_preprocessed = input_preprocessed.reindex(columns=final_cols, fill_value=0)
        
        # Make prediction
        prob = model.predict_proba(input_preprocessed)[0]
        fraud_prob = prob[1] * 100
        is_fraud = 1 if prob[1] >= threshold else 0
        
        # Show Results
        st.markdown("### 🔍 Risk Assessment Result")
        
        if is_fraud == 1: # Highlight risk if predicted fraud based on dynamic threshold
            st.markdown(f"""
            <div class='risk-high'>
                <h3 style='margin: 0 0 10px 0; color: #fca5a5;'>⚠️ HIGH FRAUD RISK WARNING</h3>
                <p style='font-size: 1.1rem; margin-bottom: 15px;'>
                    This claim exhibits patterns highly correlated with historical fraudulent cases. 
                    Manual underwriting review is strongly recommended.
                </p>
                <div style='display: flex; gap: 30px; align-items: center;'>
                    <div>
                        <div class='metric-title' style='color: #fca5a5;'>Fraud Probability</div>
                        <div class='metric-value' style='font-size: 2.5rem; color: #ffffff;'>{fraud_prob:.1f}%</div>
                    </div>
                    <div style='background-color: rgba(255,255,255,0.1); width: 2px; height: 50px;'></div>
                    <div>
                        <div class='metric-title' style='color: #fca5a5;'>Recommendation</div>
                        <div class='metric-value' style='font-size: 1.5rem; color: #ffffff;'>Flag for Investigation</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='risk-low'>
                <h3 style='margin: 0 0 10px 0; color: #a7f3d0;'>✅ LOW FRAUD RISK</h3>
                <p style='font-size: 1.1rem; margin-bottom: 15px;'>
                    This claim aligns with normal patterns and exhibits low risk of fraud.
                </p>
                <div style='display: flex; gap: 30px; align-items: center;'>
                    <div>
                        <div class='metric-title' style='color: #a7f3d0;'>Fraud Probability</div>
                        <div class='metric-value' style='font-size: 2.5rem; color: #ffffff;'>{fraud_prob:.1f}%</div>
                    </div>
                    <div style='background-color: rgba(255,255,255,0.1); width: 2px; height: 50px;'></div>
                    <div>
                        <div class='metric-title' style='color: #a7f3d0;'>Recommendation</div>
                        <div class='metric-value' style='font-size: 1.5rem; color: #ffffff;'>Standard Fast-Track</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Model Insights & Analytics")
    
    # Dynamic metric computation
    y_test = metadata['y_test']
    y_test_probs = metadata['y_test_probs']
    y_pred = (y_test_probs >= threshold).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    
    # Calculate dynamic lift (prevalence is target mean * 100)
    baseline_prevalence = y_test.mean() * 100
    lift = (prec / baseline_prevalence) if prec > 0 else 1.0
    
    # Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Audit Efficiency (Lift)</div>
            <div class='metric-value'>{lift:.1f}x</div>
            <div style='color: #64748b; font-size: 0.85rem; margin-top: 5px;'>Efficiency multiplier vs. random manual audit</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Precision (Fraud Class)</div>
            <div class='metric-value'>{prec:.1f}%</div>
            <div style='color: #64748b; font-size: 0.85rem; margin-top: 5px;'>How many flagged claims are actually fraud</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Recall (Fraud Class)</div>
            <div class='metric-value'>{rec:.1f}%</div>
            <div style='color: #64748b; font-size: 0.85rem; margin-top: 5px;'>Percentage of actual fraud cases caught</div>
        </div>
        """, unsafe_allow_html=True)

    # Plot top features
    st.markdown("#### 🔑 Top 10 Most Important Features")
    st.markdown("These are the most critical factors the model uses to determine the risk of fraud.")
    
    # Calculate feature importances
    importances = model.feature_importances_
    features = metadata['feature_columns']
    
    # Map back encoded column names to clean readable labels if possible
    # Get top 10
    indices = np.argsort(importances)[::-1][:10]
    top_features = [features[i] for i in indices]
    top_importances = importances[indices]
    
    # Create pandas dataframe for plotting
    chart_data = pd.DataFrame({
        'Feature Importance': top_importances
    }, index=top_features)
    
    st.bar_chart(chart_data)
    
    st.info(
        f"💡 **Data Science Portfolio Insight**: In fraud detection, the target class is highly imbalanced (only {baseline_prevalence:.1f}% of cases are fraud). "
        "Standard classification accuracy is a misleading metric here, as a naive model predicting 100% 'No Fraud' would achieve high accuracy but catch 0% of fraud. "
        "Therefore, we optimize for **Recall** (catching the maximum amount of fraud) and **Audit Efficiency (Lift)**. "
        f"At your selected decision threshold of **{threshold:.2f}**, the model is **{lift:.1f}x more efficient** at identifying fraud than random audits, "
        f"successfully catching **{rec:.1f}%** of all fraud cases in the test set."
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Model Details & Training Metadata")
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    with meta_col1:
        st.markdown("""
        <div class='card'>
            <div class='metric-title'>Algorithm</div>
            <div class='metric-value' style='font-size: 1.3rem;'>XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    with meta_col2:
        st.markdown("""
        <div class='card'>
            <div class='metric-title'>Class Weights</div>
            <div class='metric-value' style='font-size: 1.3rem;'>scale_pos_weight</div>
        </div>
        """, unsafe_allow_html=True)
    with meta_col3:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Training Samples</div>
            <div class='metric-value' style='font-size: 1.3rem;'>12,335</div>
        </div>
        """, unsafe_allow_html=True)
    with meta_col4:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-title'>Input Features</div>
            <div class='metric-value' style='font-size: 1.3rem;'>{len(metadata['feature_columns'])} (Encoded)</div>
        </div>
        """, unsafe_allow_html=True)
