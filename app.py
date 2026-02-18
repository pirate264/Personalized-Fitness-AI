import streamlit as st
import kagglehub
import pandas as pd
import os
import pickle
import numpy as np

# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="AI Fitness Planner",
    page_icon="💪",
    layout="centered"
)

st.title("💪 AI Personalized Diet & Exercise Planner")
st.write("Machine Learning + Kaggle Datasets + BMI Visualization")

# ======================================================
# Load Models
# ======================================================
@st.cache_resource
def load_diet_model():
    with open("diet_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_exercise_model():
    with open("exercise_model.pkl", "rb") as f:
        return pickle.load(f)

diet_model = load_diet_model()
exercise_model = load_exercise_model()

# ======================================================
# Load Diet Dataset
# ======================================================
@st.cache_data
def load_diet_dataset():
    path = kagglehub.dataset_download("batthulavinay/indian-food-nutrition")
    for file in os.listdir(path):
        if file.endswith(".csv"):
            return pd.read_csv(os.path.join(path, file))

diet_df = load_diet_dataset()
diet_df.columns = diet_df.columns.str.strip()
diet_df = diet_df.dropna()

# ======================================================
# Load Exercise Dataset
# ======================================================
@st.cache_data
def load_exercise_dataset():
    path = kagglehub.dataset_download(
        "aadhavvignesh/calories-burned-during-exercise-and-activities"
    )
    for file in os.listdir(path):
        if file.endswith(".csv"):
            return pd.read_csv(os.path.join(path, file))

exercise_df = load_exercise_dataset()
exercise_df.columns = exercise_df.columns.str.strip()
exercise_df = exercise_df.dropna()

# ======================================================
# BMI Function + Image Mapping
# ======================================================
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        return round(bmi, 2), "Slim", "https://img.freepik.com/premium-vector/cartoon-woman-lifting-weights-gym-fitness_95264-4937.jpg?w=996"
    elif bmi < 25:
        return round(bmi, 2), "Normal", "https://png.pngtree.com/png-clipart/20190117/ourlarge/pngtree-gym-fitness-motion-girl-png-image_437506.jpg"
    elif bmi < 30:
        return round(bmi, 2), "Chubby", "https://img.freepik.com/premium-vector/chubby-woman-exercise-with-dumbbell-weight-training-fitness-gym-room_294791-852.jpg?w=2000"
    else:
        return round(bmi, 2), "Obese", "https://png.pngtree.com/png-vector/20240914/ourlarge/pngtree-cartoon-fat-man-training-muscle-png-image_12932018.png"

# ======================================================
# Sidebar Inputs
# ======================================================
st.sidebar.header("👤 User Details")

height = st.sidebar.number_input("Height (cm)", 140, 210, 170)
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)

target_calories = st.sidebar.slider(
    "Daily Calorie Target (kcal)", 1200, 3500, 2000, 100
)

meal_type = st.sidebar.selectbox(
    "Meal Type", ["Breakfast", "Lunch", "Dinner"]
)

st.sidebar.header("🏃 Exercise Details")

exercise_duration = st.sidebar.slider(
    "Exercise Duration (minutes)", 10, 120, 30
)

# ======================================================
# BMI Display with Image
# ======================================================
bmi, bmi_category, bmi_image = calculate_bmi(weight, height)

st.subheader("📊 Body Mass Index (BMI)")

col1, col2 = st.columns([1, 1])

with col1:
    st.metric("BMI Value", bmi)
    st.metric("Body Type", bmi_category)

with col2:
    st.image(bmi_image, caption=f"{bmi_category} Body Type", width=220)

if bmi_category == "Slim":
    st.info("💡 Focus on strength training and calorie-dense foods.")
elif bmi_category == "Chubby":
    st.warning("💡 Increase cardio and control calorie intake.")
elif bmi_category == "Obese":
    st.error("💡 Low-impact cardio and strict calorie balance recommended.")

# ======================================================
# Diet Recommendation (ML-based)
# ======================================================
def recommend_diet(df, model, target_calories):
    df = df.copy()

    X = df[["Protein (g)", "Carbohydrates (g)", "Fats (g)"]]
    df["Predicted_Calories"] = model.predict(X)

    per_meal_cal = target_calories / 3

    df["Score"] = (
        -abs(df["Predicted_Calories"] - per_meal_cal)
        + 2 * df["Protein (g)"]
    )

    return df.sort_values("Score", ascending=False).head(5)

# ======================================================
# Exercise Recommendation (ML-based)
# ======================================================
def recommend_exercise(df, model, weight_kg, duration_min):
    df = df.copy()

    X = df[["130 lb", "155 lb", "180 lb", "205 lb"]]
    df["Predicted_Cal_per_kg"] = model.predict(X)

    duration_hr = duration_min / 60
    df["Calories_Burned"] = (
        df["Predicted_Cal_per_kg"] * weight_kg * duration_hr
    )

    return df.sort_values("Calories_Burned", ascending=False).head(5)

# ======================================================
# Generate Button
# ======================================================
generate = st.button("🚀 Generate AI Fitness Plan")

# ======================================================
# Diet Output
# ======================================================
if generate:
    st.subheader(f"🍽️ {meal_type} Diet Recommendations (ML-based)")

    diet_result = recommend_diet(
        diet_df, diet_model, target_calories
    )

    st.dataframe(
        diet_result[
            [
                "Dish Name",
                "Calories (kcal)",
                "Predicted_Calories",
                "Protein (g)",
                "Carbohydrates (g)",
                "Fats (g)",
                "Fibre (g)"
            ]
        ],
        use_container_width=True
    )

# ======================================================
# Exercise Output
# ======================================================
if generate:
    st.subheader("🏋️ Exercise Recommendations (ML-based)")

    exercise_result = recommend_exercise(
        exercise_df,
        exercise_model,
        weight,
        exercise_duration
    )

    st.dataframe(
        exercise_result[
            [
                "Activity, Exercise or Sport (1 hour)",
                "Calories_Burned"
            ]
        ],
        use_container_width=True
    )

    st.success("✅ AI Diet & Exercise plan generated successfully")

# ======================================================
# Footer
# ======================================================
st.caption(
    "📊 Datasets: Kaggle | "
    "Models: Random Forest | "
    "Features: BMI + Visual Body Type + Diet + Exercise AI"
)

