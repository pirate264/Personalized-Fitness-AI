# 💪 AI Fitness Planner  
### 🧠 Personalized Diet & Exercise Recommendation System using Machine Learning

An **AI-powered fitness planning web application** that provides **personalized diet and exercise recommendations** based on **BMI analysis**, **calorie prediction**, and **machine learning models** trained on **real-world Kaggle datasets**.  
The application is deployed using **Streamlit** and designed as a **final-year academic project**.

---

## 🚀 Key Features

✨ BMI Calculator with body-type visualization (Slim / Normal / Chubby / Obese)  
🍽️ ML-based **Diet Recommendation** using nutrition data  
🏋️ ML-based **Exercise Recommendation** with calorie burn estimation  
📊 Uses **real Kaggle datasets** via KaggleHub  
⚡ Interactive & real-time UI using Streamlit  
🎓 End-to-end ML pipeline (data → model → deployment)

---

## 🧠 System Workflow

1️⃣ User enters height, weight, calorie target & exercise duration  
2️⃣ BMI is calculated and body type is identified  
3️⃣ ML models predict:
- Calorie intake (diet)
- Calorie expenditure (exercise)  
4️⃣ Personalized diet & exercise plans are generated  
5️⃣ Results are displayed instantly in the web UI

---

## 📂 Datasets Used (Real-World Data)

### 🥗 1. Indian Food Nutrition Dataset  
📌 Source: Kaggle  
🔗 https://www.kaggle.com/datasets/batthulavinay/indian-food-nutrition  

**Description:**  
This dataset contains nutritional information of Indian dishes, including calories and macro-nutrients. It is used to train a machine learning model that predicts calorie values and recommends suitable food items.

**Key Columns:**
- Dish Name  
- Calories (kcal)  
- Protein (g)  
- Carbohydrates (g)  
- Fats (g)  
- Fibre (g)  
- Vitamins & Minerals  

---

### 🏃 2. Calories Burned During Exercise Dataset  
📌 Source: Kaggle  
🔗 https://www.kaggle.com/datasets/aadhavvignesh/calories-burned-during-exercise-and-activities  

**Description:**  
This dataset provides calorie expenditure for various physical activities based on different body weights. It is used to train an ML model for exercise calorie prediction.

**Key Columns:**
- Activity, Exercise or Sport (1 hour)  
- 130 lb  
- 155 lb  
- 180 lb  
- 205 lb  
- Calories per kg  

---

## 🧪 Machine Learning Models

🔹 **Diet Model**
- Algorithm: Random Forest Regressor  
- Inputs: Protein, Carbohydrates, Fats  
- Output: Predicted Calories  

🔹 **Exercise Model**
- Algorithm: Random Forest Regressor  
- Inputs: Activity calorie values (weight-based)  
- Output: Calories burned per kg  

📈 Models are evaluated using **MAE** and **RMSE**

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python 🐍 |
| ML Library | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Dataset Access | KaggleHub |
| Web Framework | Streamlit |
| Model Storage | Pickle |

---

## 📁 Project Structure

ai-fitness-planner/
│
├── app.py # Streamlit application
├── diet_model.pkl # Trained diet ML model
├── exercise_model.pkl # Trained exercise ML model
├── requirements.txt # Dependencies
├── README.md # Project documentation



---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
python -m streamlit run app.py
