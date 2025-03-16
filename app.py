import joblib
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# Load models
def load_baked_food_model():
    return joblib.load("model/baked_food_rf_final_v3.pkl")

def load_titanic_model():
    return joblib.load("model/titanic_model.pkl")

def load_imdb_model():
    return tf.keras.models.load_model(
        "model/imdb_rating_model.h5",
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

def load_imdb_scaler():
    return joblib.load("model/imdb_scaler.pkl")

def load_football_position_model():
    vectorizer = "model/vectorizer.pkl"
    label_encoder = "model/label_encoder.pkl"
    

    
    return vectorizer, label_encoder

# Load models
baked_food_model = load_baked_food_model()
titanic_model = load_titanic_model()
imdb_model = load_imdb_model()
imdb_scaler = load_imdb_scaler()
vectorizer, label_encoder = load_football_position_model()

# Streamlit UI
st.set_page_config(page_title="ML & NN Web App", page_icon="🎬", layout="wide")
st.title("🔍 Machine Learning & Neural Network Web App")
st.markdown("---")

menu = ["🏠 Home", "🚢 Titanic Survival Prediction", "🎥 IMDb Movie Rating Predictor", "⚽ Football Position Predictor"]
choice = st.sidebar.radio("📌 Select a Page", menu)

if choice == "🏠 Home":
    st.header("Welcome to the ML & NN Web App 🎉")
    st.write("This app showcases multiple Machine Learning models:")
    st.markdown("- **🚢 Titanic Survival Prediction**: Predict if a passenger would survive.")
    st.markdown("- **🎥 IMDb Movie Rating Predictor**: Predict IMDb ratings based on movie details.")
    st.markdown("- **⚽ Football Position Predictor**: Predict a player's most likely playing position.")

elif choice == "🚢 Titanic Survival Prediction":
    st.header("🚢 Titanic Survival Prediction")
    model = titanic_model
    
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    
    with col2:
        parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, value=50.0, step=1.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
        sex = st.radio("Gender", ["Male", "Female"])
    
    sex = 0 if sex == "Male" else 1
    embarked = {"C": 0, "Q": 1, "S": 2}[embarked]
    features = [pclass, sex, age, sibsp, parch, fare, embarked]
    
    if st.button("🔮 Predict Survival"):
        probability = model.predict_proba([features])[0][1]
        st.success(f"Prediction: {'Survived' if probability >= 0.5 else 'Not Survived'}")
        st.write(f"**Survival Probability: {probability:.2%}**")

elif choice == "🎥 IMDb Movie Rating Predictor":
    st.header("🎥 IMDb Movie Rating Predictor")
    st.subheader("Enter Movie Details to Predict IMDb Rating")
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Release Year", 1900, 2025, 2020)
        budget = st.number_input("Budget (USD)", 0, 500000000, 50000000, step=1000000)
        box_office = st.number_input("Box Office Revenue (USD)", 0, 3000000000, 100000000, step=1000000)
    
    with col2:
        run_time = st.slider("Run Time (minutes)", 30, 240, 120)
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
        st.success(f"⭐ Predicted IMDb Rating: {predicted_rating[0][0]:.2f}")

elif choice == "⚽ Football Position Predictor":
    st.header("⚽ Football Position Predictor")
    st.subheader("Enter a player's name to predict their most likely playing position.")
    
    player_name = st.text_input("Player Name", "")
    
    if st.button("🔍 Predict Position"):
        if football_position_model and vectorizer and label_encoder:
            if player_name.strip():
                player_features = vectorizer.transform([player_name])
                position_encoded = football_position_model.predict(player_features)
                predicted_position = label_encoder.inverse_transform(position_encoded)[0]
                st.success(f"🏆 Predicted Position: **{predicted_position}**")
            else:
                st.warning("⚠️ Please enter a valid player name.")
        else:
            st.error("⚠️ Model is not loaded. Please check the model files.")
