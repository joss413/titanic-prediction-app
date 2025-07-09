import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load XGBoost model
xgb_model = joblib.load('xgb_model.joblib')

st.title("🚢Titanic Survival Prediction with XGBoost")

st.write("### Enter passenger details to predict survival probability")

# User input widgets
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3], index=2)
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.number_input('Siblings/Spouses Aboard (SibSp)', 0, 8, 0)
parch = st.number_input('Parents/Children Aboard (Parch)', 0, 6, 0)
fare = st.number_input('Fare', 0.0, 600.0, 32.2, step=0.1)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'], index=2)
title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Rare'])

# Prepare input DataFrame
input_dict = {
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked],
    'Title': [title]
}
input_df = pd.DataFrame(input_dict)

# Manual encoding (match training preprocessing)
input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})
input_df['Embarked'] = input_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
input_df['Title'] = input_df['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4})

# Display input features
st.write("### Input Features")
st.write(input_df)

# Make prediction
prediction = xgb_model.predict(input_df)[0]
prediction_proba = xgb_model.predict_proba(input_df)[0][1]

st.write(f"### Predicted Survival: {'Yes' if prediction == 1 else 'No'}")
st.write(f"### Survival Probability: {prediction_proba:.2f}")

# SHAP Explainability
st.write("### SHAP Explanation")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(input_df)

# For XGBoost shap_values is usually a 2D array for binary classification
shap_values_to_plot = shap_values
base_value = explainer.expected_value

# SHAP summary plot for the input
fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values_to_plot, input_df, show=False)
st.pyplot(fig1)

# SHAP waterfall plot for the single prediction
st.write("### SHAP Waterfall Plot for This Prediction")
single_shap_values = shap_values_to_plot[0]  # First (and only) sample

expl = shap.Explanation(
    values=single_shap_values,
    base_values=base_value,
    data=input_df.iloc[0]
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
shap.plots.waterfall(expl, max_display=10, show=False)
st.pyplot(fig2)
