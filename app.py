import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import hog

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Load Model
# ------------------------------
model = load_model("brain_tumor_ann_best.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

IMG_SIZE = 128

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Project Description",
        "Prediction",
        "User Instructions",
        "About"
    ]
)

# ------------------------------
# Home
# ------------------------------
if page == "Home":

    st.title("🧠 Brain Tumor MRI Classification")

    st.image(
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=800",
        use_container_width=True,
    )

    st.markdown("## Welcome")

    st.write("""
This application predicts the type of brain tumor from MRI images using an
Artificial Neural Network (ANN).

Developed as an Image-Based ANN Capstone Project.
""")

# ------------------------------
# Project Description
# ------------------------------
elif page == "Project Description":

    st.title("📖 Project Description")

    st.subheader("Problem Statement")

    st.write("""
Develop an Artificial Neural Network (ANN) model to classify
brain MRI images into different tumor categories.
""")

    st.subheader("Business Objective")

    st.write("""
Assist healthcare professionals by automatically classifying
brain MRI images.
""")

    st.subheader("Dataset")

    st.write("""
• Brain Tumor MRI Dataset

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor
""")

    st.subheader("Technologies Used")

    st.write("""
- Python
- OpenCV
- HOG Feature Extraction
- TensorFlow / Keras
- Optuna
- Streamlit
""")

# ------------------------------
# Prediction
# ------------------------------
elif page == "Prediction":

    st.title("🔍 Brain Tumor Prediction")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_container_width=True
        )

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = gray / 255.0

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features = features.reshape(1, -1)

        features = scaler.transform(features)

        prediction = model.predict(features, verbose=0)

        predicted_index = np.argmax(prediction)

        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = np.max(prediction) * 100

        st.success(f"Prediction : {predicted_label}")

        st.info(f"Confidence Score : {confidence:.2f}%")

# ------------------------------
# User Instructions
# ------------------------------
elif page == "User Instructions":

    st.title("📋 User Instructions")

    st.markdown("""
1. Upload a Brain MRI image.

2. Supported formats:
   - JPG
   - JPEG
   - PNG

3. Wait for prediction.

4. View:
   - Tumor Type
   - Confidence Score
""")

# ------------------------------
# About
# ------------------------------
elif page == "About":

    st.title("ℹ About")

    st.write("""
### Model Information

Model : Artificial Neural Network (ANN)

Feature Extraction : HOG

Optimizer : Adam

Hyperparameter Tuning : Optuna

Deployment : Streamlit
""")