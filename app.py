import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("svr_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def main():
    st.set_page_config(page_title="SVR Housing Price Predictor", layout="centered")
    st.title("SVR Housing Price Predictor")
    st.write(
        "Predict the California housing median income using the trained SVR model."
    )

    model = load_model()
    if model is None:
        st.error("Model file not found: svr_model.pkl")
        st.stop()

    st.subheader("Input Features")
    col1, col2 = st.columns(2)

    with col1:
        house_age = st.number_input("HouseAge", min_value=0.0, value=20.0, step=0.1)
        ave_rooms = st.number_input("AveRooms", min_value=0.0, value=5.0, step=0.1)
        ave_bedrms = st.number_input(
            "AveBedrms", min_value=0.0, value=1.0, step=0.1
        )
        population = st.number_input("Population", min_value=0.0, value=1000.0, step=10.0)

    with col2:
        ave_occup = st.number_input("AveOccup", min_value=0.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.0, step=0.01)
        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0, value=-118.0, step=0.01
        )

    feature_names = [
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    input_data = pd.DataFrame(
        [[house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]],
        columns=feature_names,
    )

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted value: {prediction:.4f}")


if __name__ == "__main__":
    main()
