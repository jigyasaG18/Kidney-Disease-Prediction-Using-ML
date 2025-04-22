# Import necessary libraries
import streamlit as st  # For building the web application
import pandas as pd  # For data manipulation with DataFrames
import pickle  # For loading saved model and scaler files
import time

# Load the encoder, scaler, and trained model from saved files
scaler = pickle.load(open("scaler.pkl", 'rb'))  # Load the scaler that was used to standardize data
model_gbc = pickle.load(open("model_gbc.pkl", 'rb'))  # Load the pre-trained model for predictions

def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    # Create a DataFrame from user input data for prediction
    df_dict = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'hemo': [hemo],
        'sc': [sc],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pc': [pc]
    }
    df = pd.DataFrame(df_dict)  # Convert the dictionary to a DataFrame

    # Encode categorical variables into numerical values for the model
    df['htn'] = df['htn'].map({'yes': 1, "no": 0})
    df['dm'] = df['dm'].map({'yes': 1, "no": 0})
    df['cad'] = df['cad'].map({'yes': 1, "no": 0})
    df['appet'] = df['appet'].map({'good': 1, "poor": 0})
    df['pc'] = df['pc'].map({'normal': 1, "abnormal": 0})

    # Scale the numeric columns using the previously fitted scaler
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])  # Standardize numerical features

    # Make the prediction using the trained model
    prediction = model_gbc.predict(df)  # Get the prediction result from the model

    # Return the predicted class (0 or 1)
    return prediction[0]

# Sidebar for additional information
st.sidebar.title("About This App")
st.sidebar.markdown("""
    This application is designed to predict Chronic Kidney Disease (CKD) using patient data.
    Enter the necessary parameters to get the prediction results.
""")

st.sidebar.title("Usefulness")
st.sidebar.markdown("""
    The app helps healthcare professionals identify patients at risk of kidney disease,
    enabling proactive management and timely intervention.
""")

st.sidebar.title("Use Cases")
st.sidebar.markdown("""
- **Healthcare Settings**: Used by doctors to assess kidney health in patients.
- **Research**: Useful for studies related to kidney function and disease patterns.
- **Patient Education**: Helps patients understand their health conditions.
""")

st.sidebar.title("Future Advancements")
st.sidebar.markdown("""
    - Integration with Electronic Health Records (EHR) for real-time data analysis.
    - Machine learning enhancements for improved prediction accuracy.
    - User-friendly interface upgrades for better patient engagement.
""")

# Streamlit UI setup
st.title('Chronic Kidney Disease Prediction')  # Set the title for the web app

# Display the image
st.image("kidney.jpg", use_container_width=True)  # Adjust 'use_column_width' as needed

# Create two columns for input layout
col1, col2 = st.columns(2)

with col1:
    # Input fields for the user to enter data
    age = st.number_input("Age", min_value=1, max_value=120, value=48)  # Age input
    bp = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)  # Blood pressure input
    sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.050, value=1.020)  # Specific gravity input
    al = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=1.0)  # Albumin level input
    hemo = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=15.4)  # Hemoglobin level input
    sc = st.number_input("Serum Creatinine", min_value=0.5, max_value=10.0, value=1.2)  # Serum creatinine input

with col2:
    # Dropdown menus for selecting conditions
    htn = st.selectbox("Hypertension", ["yes", 'no'])  # Hypertension input
    dm = st.selectbox("Diabetes", ["yes", 'no'])  # Diabetes input
    cad = st.selectbox("Coronary Artery Disease", ["yes", 'no'])  # Coronary artery disease input
    appet = st.selectbox("Appetite", ["good", "poor"])  # Appetite condition input
    pc = st.selectbox("Protein in Urine", ["normal", "abnormal"])  # Protein level input

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Show a loading spinner while processing
    with st.spinner('Generating prediction...'):
        time.sleep(2)  # Simulate a delay if necessary

        # Make the prediction
        result = predict_chronic_disease(age,bp,sg,al,hemo,sc,htn,dm,cad,appet,pc)
        
        # Display the result with messages
        if result == 1:
            st.success("🚨 The patient has Chronic Kidney Disease (CKD).")  # Green success message
        else:
            st.success("✅ The patient does not have Chronic Kidney Disease (CKD).")  # Green success message
