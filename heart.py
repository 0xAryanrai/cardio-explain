import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import datetime # <-- NEW: We imported this to put a timestamp on the report

# --- SET WIDE PAGE LAYOUT ---
st.set_page_config(layout="wide", page_title="Heart Disease Predictor")

# ---------------------------------------------------------
# 1. Initialize "Memory" (Session State) & Callbacks
# ---------------------------------------------------------
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []

def delete_patient(index):
    st.session_state.patient_history.pop(index)

# ---------------------------------------------------------
# 2. Load the "Brain" of the App
# ---------------------------------------------------------
model = joblib.load('best_heart_disease_model.pkl')
features = joblib.load('model_features.pkl')

st.title("🫀 Heart Disease Prediction Dashboard")
st.write("Enter patient details to assess risk. Previous assessments are saved in the history panel.")
st.markdown("---")

# ---------------------------------------------------------
# 3. Build the User Interface (3 Columns)
# ---------------------------------------------------------
col1, col2, col3 = st.columns([2, 2, 1.5])

with col1:
    st.header("Basic Details")
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    thalch = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [True, False])

with col2:
    st.header("Clinical Tests")
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    restecg = st.selectbox("Resting ECG Results", ["normal", "lv hypertrophy", "st-t abnormality"])
    exang = st.selectbox("Exercise Induced Angina?", [True, False])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels", [0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

with col3:
    st.header("📋 Patient History")
    if len(st.session_state.patient_history) == 0:
        st.info("No records yet. Predict a patient's risk to save it here.")
    
    for index, record in enumerate(st.session_state.patient_history):
        with st.expander(f"Patient {index + 1} | Age: {record['age']}, {record['sex']}"):
            if record['high_risk']:
                st.error(f"⚠️ High Risk ({record['risk_score']:.1f}%)")
            else:
                st.success(f"✅ Low Risk ({100 - record['risk_score']:.1f}%)")
            
            st.markdown(f"""
            **Vitals:**
            * BP: {record['trestbps']} | Chol: {record['chol']}
            * Max HR: {record['thalch']} | FBS > 120: {record['fbs']}
            
            **Clinical:**
            * Chest Pain: {record['cp']}
            * ECG: {record['restecg']}
            * Angina: {record['exang']}
            
            **Advanced:**
            * ST Dep: {record['oldpeak']} | Slope: {record['slope']}
            * Vessels: {record['ca']} | Thal: {record['thal']}
            """)
            
            st.button("🗑️ Delete Record", key=f"del_{index}", on_click=delete_patient, args=(index,))

# ---------------------------------------------------------
# 4. The Prediction Engine & Memory Save
# ---------------------------------------------------------
if st.button("Predict Heart Disease Risk", type="primary"):
    
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)
    input_2d = np.array(input_encoded).reshape(1, -1)
    
    prediction = model.predict(input_2d)[0]
    probability = model.predict_proba(input_2d)[0][1] * 100
    
    new_record = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
        'high_risk': bool(prediction == 1),
        'risk_score': probability
    }
    st.session_state.patient_history.append(new_record)
    st.rerun()

# ---------------------------------------------------------
# 5. Dashboard Output 
# ---------------------------------------------------------
if len(st.session_state.patient_history) > 0:
    st.markdown("---")
    st.header("Latest Diagnosis Results")
    
    latest_prob = st.session_state.patient_history[-1]['risk_score']
    latest_risk = st.session_state.patient_history[-1]['high_risk']
    
    col_chart, col_shap = st.columns(2)
    
    with col_chart:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk Level (%)", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "gold"},
                    {'range': [70, 100], 'color': "tomato"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_shap:
        st.subheader("🧠 Model Explanation")
        latest_patient_data = pd.DataFrame([{
            k: st.session_state.patient_history[-1][k] for k in 
            ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        }])
        
        input_encoded = pd.get_dummies(latest_patient_data).reindex(columns=features, fill_value=0)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_encoded)
        
        if isinstance(shap_values, list):
            patient_shap_values = shap_values[1][0] 
        elif len(np.shape(shap_values)) == 3:
            patient_shap_values = shap_values[0, :, 1]
        else:
            patient_shap_values = shap_values[0]
            
        shap_df = pd.DataFrame({"Impact on Risk Score": patient_shap_values}, index=input_encoded.columns)
        shap_df['Absolute Impact'] = shap_df["Impact on Risk Score"].abs()
        shap_df = shap_df.sort_values(by="Absolute Impact", ascending=False).drop(columns=["Absolute Impact"])
        
        st.bar_chart(shap_df.head(6))

    st.markdown("---")
    
    # ---------------------------------------------------------
    # 6. DOWNLOAD PATIENT REPORT FUNCTION
    # ---------------------------------------------------------
    st.subheader("📄 Export Patient Data")
    st.write("Generate a secure text report for this patient's medical file.")
    
    # Grab the latest patient's info from memory
    latest_patient = st.session_state.patient_history[-1]
    
    # Determine the text for the diagnosis
    diagnosis_text = "HIGH RISK" if latest_patient['high_risk'] else "LOW RISK"
    
    # Create the formatted text document
    report_content = f"""
========================================
    AI CLINICAL CARDIOLOGY REPORT       
========================================
Date Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

--- PATIENT DEMOGRAPHICS ---
Age: {latest_patient['age']}
Sex: {latest_patient['sex']}

--- VITAL SIGNS ---
Resting Blood Pressure: {latest_patient['trestbps']} mm Hg
Cholesterol: {latest_patient['chol']} mg/dl
Max Heart Rate Achieved: {latest_patient['thalch']} bpm
Fasting Blood Sugar > 120 mg/dl: {latest_patient['fbs']}

--- CLINICAL EVALUATION ---
Chest Pain Type: {latest_patient['cp']}
Resting ECG: {latest_patient['restecg']}
Exercise Induced Angina: {latest_patient['exang']}
ST Depression (Oldpeak): {latest_patient['oldpeak']}
Slope of ST Segment: {latest_patient['slope']}
Number of Major Vessels: {latest_patient['ca']}
Thalassemia: {latest_patient['thal']}

--- AI DIAGNOSIS ---
Prediction: {diagnosis_text}
Confidence Score: {latest_patient['risk_score']:.1f}%

Note: This report is generated by a Machine Learning model (Random Forest) 
and should not replace a comprehensive medical evaluation by a licensed cardiologist.
========================================
"""

    # Create the download button
    st.download_button(
        label="⬇️ Download Clinical Report (.txt)",
        data=report_content,
        file_name=f"patient_report_{latest_patient['age']}{latest_patient['sex'][0]}.txt",
        mime="text/plain"
    )