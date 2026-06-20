import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# --- HOME ROUTE ---
@app.route('/')
def home():
    # Automatically redirects you to the breast cancer page when you open the site
    return redirect(url_for('predict_breast'))

# --- BREAST CANCER ROUTE ---
# --- BREAST CANCER ROUTE ---
@app.route('/predict/breast', methods=['GET', 'POST'])
def predict_breast():
    result = None
    fields = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
    
    if request.method == 'POST':
        try:
            # 1. Gather values from the form
            raw_features = [float(request.form.get(field)) for field in fields]
            
            # 2. Convert into a DataFrame with proper column names
            input_df = pd.DataFrame([raw_features], columns=fields)
            
            # 3. Load the model bundle
            with open('models/breast_cancer.pkl', 'rb') as f:
                bundle = pickle.load(f)
                
            pipeline = bundle['pipeline']
            classes = bundle['classes']
            
            # 4. Predict
            prediction_idx = pipeline.predict(input_df)[0]
            actual_class = classes[prediction_idx] # This returns 1 or 2
            
            # 5. Map the 1 and 2 to your human-readable strings
            class_mapping = {1: "Healthy Controls", 2: "Patients"}
            readable_result = class_mapping.get(actual_class, f"Unknown Class ({actual_class})")
            
            result = f"Assessment Result: {readable_result}"
            
        except Exception as e:
            result = f"Error during evaluation: {str(e)}"
            
    return render_template('predict_breast.html', fields=fields, result=result)

# --- PROSTATE CANCER ROUTE ---
@app.route('/predict/prostate', methods=['GET', 'POST'])
def predict_prostate():
    result = None
    
    # Text mapping dictionaries matching your training logic
    binary_map = {"Yes": 1, "No": 0}
    alcohol_map = {"Low": 0, "Moderate": 1, "High": 2, "Unknown": -1}
    diet_map = {"Healthy": 0, "Mixed": 1, "Fatty": 2}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    
    if request.method == 'POST':
        try:
            # 1. Parse and encode raw inputs safely from the form mapping
            raw_features = {
                "age": float(request.form.get("age")),
                "bmi": float(request.form.get("bmi")),
                "smoker": binary_map[request.form.get("smoker")],
                "alcohol_consumption": alcohol_map[request.form.get("alcohol_consumption")],
                "diet_type": diet_map[request.form.get("diet_type")],
                "physical_activity_level": activity_map[request.form.get("physical_activity_level")],
                "family_history": binary_map[request.form.get("family_history")],
                "mental_stress_level": stress_map[request.form.get("mental_stress_level")],
                "sleep_hours": float(request.form.get("sleep_hours")),
                "regular_health_checkup": binary_map[request.form.get("regular_health_checkup")],
                "prostate_exam_done": binary_map[request.form.get("prostate_exam_done")]
            }
            
            # 2. Load the pipeline bundle
            with open('models/prostate_cancer.pkl', 'rb') as f:
                bundle = pickle.load(f)
                
            model = bundle['model']
            scaler = bundle['scaler']
            feature_order = bundle['features']
            
            # 3. Convert to DataFrame matching the exact trained column arrangement
            input_df = pd.DataFrame([raw_features], columns=feature_order)
            
            # 4. Scale inputs using the loaded fitted scaler
            input_scaled = scaler.transform(input_df)
            
            # 5. Predict and map output integers back to human strings
            prediction = model.predict(input_scaled)[0]
            risk_output = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            
            result = f"Assessment Risk Level: {risk_output.get(prediction, 'Unknown')}"
            
        except Exception as e:
            result = f"Error during evaluation: {str(e)}"
            
    return render_template('predict_prostate.html', result=result)

# --- LEUKEMIA ROUTE ---
@app.route('/predict/leukemia', methods=['GET', 'POST'])
def predict_leukemia():
    result = None
    
    if request.method == 'POST':
        try:
            # 1. Load the model bundle
            with open('models/leukemia.pkl', 'rb') as f:
                bundle = pickle.load(f)
                
            model = bundle['model']
            preprocessor = bundle['preprocessor']
            feature_names = bundle['feature_names']
            classes = bundle['label_encoder_classes']
            country_map = bundle['country_map']
            ethnicity_map = bundle['ethnicity_map']

            # 2. Handle manual frequency conversion for high-cardinality values
            user_country = request.form.get("Country")
            user_ethnicity = request.form.get("Ethnicity")
            
            country_freq = country_map.get(user_country, 0.0)
            ethnicity_freq = ethnicity_map.get(user_ethnicity, 0.0)

            # 3. Construct the exact dictionary structure matching raw training features (X_raw)
            # The structure order doesn't matter here because ColumnTransformer filters by column name matching
            raw_features = {
                "Age": float(request.form.get("Age")),
                "WBC_Count": float(request.form.get("WBC_Count")),
                "RBC_Count": float(request.form.get("RBC_Count")),
                "Platelet_Count": float(request.form.get("Platelet_Count")),
                "Hemoglobin_Level": float(request.form.get("Hemoglobin_Level")),
                "Bone_Marrow_Blasts": float(request.form.get("Bone_Marrow_Blasts")),
                "BMI": float(request.form.get("BMI")),
                "Country_freq": country_freq,
                "Ethnicity_freq": ethnicity_freq,
                "Socioeconomic_Status": request.form.get("Socioeconomic_Status"),
                "Gender": request.form.get("Gender"),
                "Genetic_Mutation": request.form.get("Genetic_Mutation"),
                "Family_History": request.form.get("Family_History"),
                "Smoking_Status": request.form.get("Smoking_Status"),
                "Alcohol_Consumption": request.form.get("Alcohol_Consumption"),
                "Radiation_Exposure": request.form.get("Radiation_Exposure"),
                "Infection_History": request.form.get("Infection_History"),
                "Chronic_Illness": request.form.get("Chronic_Illness"),
                "Immune_Disorders": request.form.get("Immune_Disorders"),
                "Urban_Rural": request.form.get("Urban_Rural")
            }
            
            # Convert single sample into a structured DataFrame
            input_df = pd.DataFrame([raw_features])
            
            # 4. Route features through the identical saved transformer pipeline
            transformed_features = preprocessor.transform(input_df)
            input_matrix = pd.DataFrame(transformed_features, columns=feature_names)
            
            # 5. Predict and map output
            prediction_idx = model.predict(input_matrix)[0]
            result = f"Assessment Result: {classes[prediction_idx]}"
            
        except Exception as e:
            result = f"Error during evaluation: {str(e)}"
            
    return render_template('predict_leukemia.html', result=result)
# --- CRITICAL: This keeps the server running ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)