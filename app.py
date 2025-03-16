import joblib
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# โหลดโมเดลต่าง ๆ

def load_titanic_model():
    return joblib.load("model/titanic_model.pkl")

def load_titanic_encoders():
    return joblib.load("model/titanic_label_encoders.pkl")

def load_imdb_model():
    return tf.keras.models.load_model(
        "model/imdb_rating_model.h5",
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

def load_imdb_scaler():
    return joblib.load("model/imdb_scaler.pkl")

def load_thyroid_model():
    return joblib.load("model/thyroid_model.pkl")  # ใช้โมเดลไทรอยด์

def load_thyroid_encoders():
    return joblib.load("model/thyroid_label_encoders.pkl")  # ใช้ Label Encoders

# โหลดโมเดล
titanic_model = load_titanic_model()
titanic_encoders = load_titanic_encoders()
imdb_model = load_imdb_model()
imdb_scaler = load_imdb_scaler()
thyroid_model = load_thyroid_model()  # โหลดโมเดลไทรอยด์
thyroid_encoders = load_thyroid_encoders()  # โหลด Label Encoders

# Streamlit UI
st.set_page_config(page_title="ML & NN Web App", page_icon="🔬", layout="wide")
st.title("🔍 Machine Learning & Neural Network Web App")
st.markdown("---")

menu = ["🏠 Home", "🎥 (NN)IMDb Movie Rating Predictor", "🦋 (ML)Thyroid Disease Prediction", "🚢 (ML)Titanic Survival Prediction"]
choice = st.sidebar.radio("📌 Select a Page", menu)

if choice == "🏠 Home":
    st.header("Welcome to the ML & NN Web App 🎉")
    st.write("This app showcases multiple Machine Learning models:")
    st.markdown("- **🎥 (NN)IMDb Movie Rating Predictor**: Predict IMDb ratings based on movie details.")
    st.markdown("- **🦋 (ML)Thyroid Disease Prediction**: Predict thyroid disease based on medical attributes.")

elif choice == "🎥 (NN)IMDb Movie Rating Predictor":
    st.header("IMDb Movie Rating Predictor")
    st.markdown("### Enter Movie Details to Predict IMDb Rating")
    
    year = st.number_input("Release Year", 1900, 2025, 2020)
    budget = st.number_input("Budget (USD)", 0, 500000000, 50000000)
    box_office = st.number_input("Box Office Revenue (USD)", 0, 3000000000, 100000000)
    run_time = st.number_input("Run Time (minutes)", 30, 240, 120)
    genre_main = st.selectbox("Main Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"])
    certificate = st.selectbox("Age Rating", ["G", "PG", "PG-13", "R", "NC-17"])

    genre_mapping = {"Action": 0, "Comedy": 1, "Drama": 2, "Horror": 3, "Sci-Fi": 4, "Romance": 5}
    certificate_mapping = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4}

    genre_main = genre_mapping[genre_main]
    certificate = certificate_mapping[certificate]

    if st.button("🎬 Predict IMDb Rating"):
        movie_features = np.array([[year, budget, box_office, run_time, genre_main, certificate]])
        movie_features_scaled = imdb_scaler.transform(movie_features)
        predicted_rating = imdb_model.predict(movie_features_scaled)
        st.success(f"Predicted IMDb Rating: ⭐ {predicted_rating[0][0]:.2f}")
        
elif choice == "🦋 (ML)Thyroid Disease Prediction":
    st.header("🦋 Thyroid Disease Prediction")
    
    # รับค่าจากผู้ใช้
    age = st.slider("Age", 18, 80, 30)
    gender = st.radio("Gender", ["M", "F"])
    smoking = st.radio("Smoking", ["Yes", "No"])
    hx_smoking = st.radio("History of Smoking", ["Yes", "No"])
    hx_radiotherapy = st.radio("History of Radiotherapy", ["Yes", "No"])
    thyroid_function = st.selectbox("Thyroid Function", ["Euthyroid", "Clinical Hyperthyroidism"])

    # ใช้ Encoders แปลงค่าให้ตรงกับค่าที่ใช้ตอน Train
    if gender in thyroid_encoders["Gender"].classes_:
        gender = thyroid_encoders["Gender"].transform([gender])[0]
    else:
        gender = 0 if gender == "M" else 1

    if smoking in thyroid_encoders["Smoking"].classes_:
        smoking = thyroid_encoders["Smoking"].transform([smoking])[0]
    else:
        smoking = 1 if smoking == "Yes" else 0

    if hx_smoking in thyroid_encoders["Hx Smoking"].classes_:
        hx_smoking = thyroid_encoders["Hx Smoking"].transform([hx_smoking])[0]
    else:
        hx_smoking = 1 if hx_smoking == "Yes" else 0

    if hx_radiotherapy in thyroid_encoders["Hx Radiothreapy"].classes_:
        hx_radiotherapy = thyroid_encoders["Hx Radiothreapy"].transform([hx_radiotherapy])[0]
    else:
        hx_radiotherapy = 1 if hx_radiotherapy == "Yes" else 0

    if thyroid_function in thyroid_encoders["Thyroid Function"].classes_:
        thyroid_function = thyroid_encoders["Thyroid Function"].transform([thyroid_function])[0]
    else:
        thyroid_mapping = {"Euthyroid": 2, "Clinical Hyperthyroidism": 1}
        thyroid_function = thyroid_mapping.get(thyroid_function, 0)

    # จัดข้อมูลให้เป็นรูปแบบที่โมเดลต้องการ
    features = np.array([[age, gender, smoking, hx_smoking, hx_radiotherapy, thyroid_function]])

    # ทำการทำนาย
    if st.button("🔮 Predict Thyroid Disease"):
        prediction = thyroid_model.predict(features)
        st.success(f"Prediction: {prediction[0]}")
        
elif choice == "🚢 (ML)Titanic Survival Prediction":
    st.header("🚢 Titanic Survival Prediction")
    st.markdown("### Enter Passenger Details to Predict Survival")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 1, 100, 30)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare Price", 0, 500, 50)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # แปลงค่าด้วย Encoders
    sex = titanic_encoders["Sex"].transform([sex])[0]
    embarked = titanic_encoders["Embarked"].transform([embarked])[0]

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    if st.button("🛟 Predict Survival"):
        prediction = titanic_model.predict(features)
        result = "Survived 🟢" if prediction[0] == 1 else "Not Survived 🔴"
        st.success(f"Prediction: {result}")

