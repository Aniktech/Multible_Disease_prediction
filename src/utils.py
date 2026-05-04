import shap
import matplotlib.pyplot as plt
import streamlit as st

def personal_prediction_explanation(model, input_df):
    # SHAP works best with the base estimators of the stacking model
    # We'll use the Random Forest component for the explanation
    rf_model = model.named_estimators_['rf']
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_df)

    # Create the plot
    fig, ax = plt.subplots()
    # If binary classification, index 1 is usually the 'Positive' class
    shap.summary_plot(shap_values[1], input_df, plot_type="bar", show=False)
    
    st.pyplot(fig)