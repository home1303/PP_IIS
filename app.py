import joblib
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# โหลดโมเดลต่าง ๆ
def load_titanic_model():
    return joblib.load("model/titanic_model.pkl")

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
imdb_model = load_imdb_model()
imdb_scaler = load_imdb_scaler()
thyroid_model = load_thyroid_model()  # โหลดโมเดลไทรอยด์
thyroid_encoders = load_thyroid_encoders()  # โหลด Label Encoders

# Streamlit UI
st.set_page_config(page_title="ML & NN Web App", page_icon="🔬", layout="wide")
st.title("🔍 Machine Learning & Neural Network Web App")
st.markdown("---")

menu = ["🏠 Home", "🚢 (ML)Titanic Survival Prediction", "🎥 (NN)IMDb Movie Rating Predictor", "🦋 (ML)Thyroid Disease Prediction"]
choice = st.sidebar.radio("📌 Select a Page", menu)

if choice == "🏠 Home":
    st.header("Welcome to the ML & NN Web App 🎉")
    st.write("This app showcases multiple Machine Learning models:")
    st.markdown("- **🚢 (ML)Titanic Survival Prediction**: Predict if a passenger would survive.")
    st.markdown("- **🎥 (NN)IMDb Movie Rating Predictor**: Predict IMDb ratings based on movie details.")
    st.markdown("- **🦋 (ML)Thyroid Disease Prediction**: Predict thyroid disease based on medical attributes.")

elif choice == "🚢 (ML)Titanic Survival Prediction":
    st.header("Titanic Survival Prediction")
    model = titanic_model
    features = [
        st.number_input("Pclass (1, 2, 3)", value=3),
        st.number_input("Age", value=30),
        st.number_input("SibSp", value=0),
        st.number_input("Parch", value=0),
        st.number_input("Fare", value=50),
        st.selectbox("Embarked", ["C", "Q", "S"]),
        st.selectbox("Sex", ["male", "female"])
    ]
    
    sex = 0 if features[6] == "male" else 1
    embarked = {"C": 0, "Q": 1, "S": 2}[features[5]]
    features = [features[0], sex, features[1], features[2], features[3], features[4], embarked]
    
    if st.button("Predict"): 
        probability = model.predict_proba([features])[0][1]
        st.write(f"Prediction: {'Survived' if probability >= 0.5 else 'Not Survived'}")
        st.write(f"Survival Probability: {probability:.2%}")

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
    gender = thyroid_encoders["Gender"].transform([gender])[0]
    smoking = thyroid_encoders["Smoking"].transform([smoking])[0]
    hx_smoking = thyroid_encoders["Hx Smoking"].transform([hx_smoking])[0]
    hx_radiotherapy = thyroid_encoders["Hx Radiothreapy"].transform([hx_radiotherapy])[0]
    thyroid_function = thyroid_encoders["Thyroid Function"].transform([thyroid_function])[0]

    # จัดข้อมูลให้เป็นรูปแบบที่โมเดลต้องการ
    features = np.array([[age, gender, smoking, hx_smoking, hx_radiotherapy, thyroid_function]])

    # ทำการทำนาย
    if st.button("🔮 Predict Thyroid Disease"):
        prediction = thyroid_model.predict(features)
        st.success(f"Prediction: {prediction[0]}")
