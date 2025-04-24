# Kidney Disease Prediction Using ML

## Overview
Chronic Kidney Disease (CKD) presents a significant health challenge globally, affecting millions and leading to severe complications, including end-stage kidney failure. Early detection and intervention can halt or slow disease progression, making prediction systems invaluable in clinical settings. This project leverages machine learning algorithms to create a predictive model for CKD. By analyzing multiple patient attributes, the model aims to provide healthcare practitioners with a powerful tool to identify high-risk patients and facilitate timely medical interventions.

## Table of Contents
- [Introduction](#introduction)
- [Process Overview](#process-overview)
- [Dataset Description](#dataset-description)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Encoding Categorical Variables](#encoding-categorical-variables)
  - [Scaling Features](#scaling-features)
- [Balancing the Dataset](#balancing-the-dataset)
- [Model Training](#model-training)
  - [Classifier Selection](#classifier-selection)
- [Model Evaluation](#model-evaluation)
- [Best Model Selection](#best-model-selection)
- [Model Persistence (Saving the Model)](#model-persistence-saving-the-model)
- [Inference for New Data](#inference-for-new-data)
- [User Interface (Optional)](##user-interface-optional)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction
The kidneys play a critical role in maintaining the body’s fluid, electrolyte balance and eliminating waste products. When their functionality diminishes due to various factors, including hypertension and diabetes, it can lead to chronic kidney disease (CKD). According to the World Health Organization (WHO), the prevalence of CKD has been rising steadily, necessitating robust solutions for early detection.

Machine learning, a subset of artificial intelligence (AI), has shown considerable promise in health-related predictive analytics. This project aims to develop a comprehensive machine learning pipeline to predict CKD, integrating a variety of classifiers and evaluating their performance to select the most effective one.

## Process Overview
The project follows a structured process that encompasses multiple stages, ensuring thorough data handling and model performance evaluation. The principal steps in developing the kidney disease prediction model are outlined below:

1. **Imports**: Load necessary libraries and frameworks essential for data handling and modeling.
2. **Load Dataset**: Read the CSV file containing the patient data and inspect its structure.
3. **Data Cleaning**: Clean the dataset by handling missing values, removing tabs, and ensuring consistency in formatting.
4. **Fill Missing Values**: Impute missing data using statistical techniques appropriate for their types (categorical and numerical).
5. **Encoding**: Convert categorical variables into numerical format, facilitating better handling by the algorithms.
6. **Scaling**: Normalize numerical features to ensure consistent data distribution.
7. **Data Balancing**: Apply techniques to address class imbalances in the dataset, ensuring that the model treats both positive and negative classes equally.
8. **Train-Test Split**: Split the dataset into training and testing sets to validate the model's effectiveness.
9. **Train & Evaluate Multiple Classifiers**: Implement various machine learning algorithms and evaluate their performance.
10. **Select Best Model**: Identify the classifier that yields the best performance metrics.
11. **Save Model & Scaler**: Persist the chosen model and pre-processing steps for future use.
12. **Inference**: Create a prediction function to assess new patient data for CKD risk.

## Dataset Description
The dataset utilized in this project includes extensive information regarding patient health metrics related to kidney function. The characteristics of the dataset include:

- **Age**: Continuous variable representing the patient's age in years.
- **Blood Pressure (bp)**: Continuous variable measuring blood pressure in mm/Hg.
- **Specific Gravity (sg)**: Continuous variable indicating urine specific gravity.
- **Albumin (al)**: Continuous variable measuring the level of albumin in urine.
- **Hemoglobin (hemo)**: Continuous variable representing hemoglobin concentration in grams per deciliter.
- **Serum Creatinine (sc)**: Continuous variable indicating serum creatinine levels, a key marker of kidney function.
- **Hypertension (htn)**: Categorical variable indicating if the patient has hypertension (yes/no).
- **Diabetes Mellitus (dm)**: Categorical variable indicating if the patient has diabetes (yes/no).
- **Coronary Artery Disease (cad)**: Categorical variable indicating if the patient has coronary artery disease (yes/no).
- **Appetite (appet)**: Categorical variable indicating appetite status (good/poor).
- **Pus Cell Status (pc)**: Categorical variable indicating presence of pus cells in urine (normal/abnormal).
- **Classification**: Target variable indicating CKD diagnosis (ckd/notckd), with ‘ckd’ being the positive class.

The dataset consists of 400 patient records across 12 attributes, making it a well-structured collection of valuable health information.

## Data Cleaning and Preprocessing
### Handling Missing Values
The first step in data cleaning is addressing missing values. Missing data can skew the model's predictions and impact the validity of results. Here, values are imputed using:

- **Numerical Features**: Filled with the median value of the respective columns.
- **Categorical Features**: Filled using the mode of each column.

The outcome is a complete dataset ready for further analysis without any missing values.

### Encoding Categorical Variables
Machine learning algorithms operate primarily with numerical inputs. Thus, it is essential to convert categorical variables into a numerical format. Various encoding strategies were utilized, such as:

- **Label Encoding**: Categorical values such as "yes" and "no" are converted to binary values (1 and 0).
- **Mapping**: Other categorical variables (like `appet` and `pc`) are similarly mapped to numerical representations, resulting in a dataset that can be effectively processed by algorithms.

### Scaling Features
Scaling involves normalizing numerical variables to ensure that they all contribute equally to the model's predictions. This is achieved through Min-Max scaling, transforming the data into a range between 0 and 1 using the formula:

{scaled_value} = {x - {min}}{{max} - {min}}

This technique improves the performance of machine learning algorithms by maintaining consistent data distributions.

## Balancing the Dataset
Data balancing is crucial when working with classification tasks, particularly when one class significantly outnumbers another. This project uses Synthetic Minority Over-sampling Technique (SMOTE) to address this issue. SMOTE generates synthetic samples for the minority class (in this case, patients with CKD), thereby balancing the number of samples across all classes. The balanced dataset ensures that the model does not become biased toward the majority class.

## Model Training
### Classifier Selection
Various machine learning models were selected for training, allowing for a comparative analysis of their effectiveness. The classifiers included:

- **Logistic Regression**: A foundational model effective for binary classification.
- **Support Vector Classifier (SVC)**: A powerful model employing hyperplane separation and kernels for non-linear cases.
- **Random Forest Classifier**: An ensemble model using multiple decision trees for robust predictions.
- **K Nearest Neighbors (KNN)**: A distance-based model where predictions are based on the nearest neighbors.
- **Decision Tree Classifier**: A straightforward tree-based model that mimics human decision-making.
- **Gaussian Naive Bayes**: A probabilistic model based on Bayes’ theorem.
- **AdaBoost Classifier**: An ensemble technique that combines weak classifiers to create a strong learner.
- **Gradient Boosting Classifier**: Another ensemble method focusing on preventing errors of basic models through sequential training.

Each model is trained and subsequently evaluated against the test set to gauge predictive performance.

## Model Evaluation
Model performance is assessed using several key metrics, including:

- **Accuracy**: The proportion of true results, showing how many predictions were correct.
- **Precision**: The fraction of correct positive predictions over total positive predictions, crucial for imbalanced datasets.
- **Recall**: Also known as sensitivity, this metric indicates how many actual positive cases were captured by the model.
- **F1-Score**: The harmonic mean of precision and recall, offering a balance between the two metrics.
- **Confusion Matrix**: A table that visualizes true vs. predicted classifications, highlighting the model's strengths and weaknesses.

After evaluating all classifiers, their results provide insight into which model performs best and under what conditions.

## Best Model Selection
Through comparative analysis, the Gradient Boosting Classifier emerged as the top-performing model, yielding high accuracy and strong F1-scores. The classifier's ability to manage data complexity and its resistance to overfitting contributed to its effectiveness. A confusion matrix and classification report further validate its predictive power:

```python
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
```

## Model Persistence (Saving the Model)
Once the model is finalized, it is imperative to save both the trained model and the scaler to make future predictions easier. Using the `pickle` library, the model and scaler are stored:

```python
import pickle
import os

# Create the directory to store models if it does not exist
os.makedirs('models', exist_ok=True)

# Save the scaler and model
pickle.dump(scaler, open("models/scaler.pkl", 'wb'))
pickle.dump(model_gbc, open("models/model_gbc.pkl", 'wb'))
```

This encapsulation allows developers to retrieve the model and scaler without redundant re-training efforts.

## Inference for New Data
Prediction on new patient data involves several steps, mirroring the preprocessing pipeline followed during training. The loading of the scaler and model is essential for transforming incoming data correctly:

```python
# Load the trained model and scaler
scaler = pickle.load(open("models/scaler.pkl", 'rb'))
model_gbc = pickle.load(open("models/model_gbc.pkl", 'rb'))

def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    # Create a new DataFrame for input features
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
    df = pd.DataFrame(df_dict)

    # Apply categorical encoding
    df['htn'] = df['htn'].map({'yes':1, "no":0})
    df['dm'] = df['dm'].map({'yes':1, "no":0})
    df['cad'] = df['cad'].map({'yes':1, "no":0})
    df['appet'] = df['appet'].map({'good':1, "poor":0})
    df['pc'] = df['pc'].map({'normal':1, "abnormal":0})

    # Scale the features
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Generate prediction
    prediction = model_gbc.predict(df)

    return prediction[0]  # Return the predicted class
```

This function facilitates the quick and easy generation of predictions for new patients, streamlining the clinical workflow.

## User Interface
To enhance usability, a simple user interface has been developed using Streamlit framework. Such a GUI would allow medical practitioners or patients to input their health parameters and receive immediate feedback regarding their CKD risk. This feature can significantly promote awareness and enable proactive health management.

## Conclusion and Future Work
This project demonstrates the impactful integration of machine learning in healthcare, specifically in predicting chronic kidney disease. The developed model, with high accuracy and reliability, presents a promising tool for early detection and intervention in CKD patients. 

In the future, this system could be improved through:
- **Integration with Electronic Health Records (EHR)**: Automating data retrieval and predictions within hospital systems.
- **Extended Data Collection**: Including additional variables, such as lifestyle factors, that may impact kidney health.
- **Longitudinal Studies**: Implementing tracking of patients over time to assess the model's real-world effectiveness.
- **Public Awareness Campaigns**: Using this model to educate populations at risk regarding CKD and preventative measures.

Ultimately, further development and deployment of this predictive model hold the potential to substantially impact patient outcomes and public health as a whole. This project exemplifies how advanced analytics can shape the future of medical diagnostics and improve patient care.
