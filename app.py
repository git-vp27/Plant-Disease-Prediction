import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import warnings
import os

warnings.filterwarnings("ignore")

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# =============================
# ✅ Load Image Model
# =============================
@st.cache_resource
def load_image_model():
    model_path = "PD_New_model.h5"  # Ensure this file exists in the same directory
    if not os.path.exists(model_path):
        st.error("❌ Model file 'PD_model.h5' not found. Please upload or place it in the same directory.")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_image_model()

# =============================
# ✅ Class Labels
# =============================
class_labels = [
    "Apple_Black_Rot", "Apple_Healthy", "Apple_Rust", "Apple_Scab",
    "Cherry_Healthy", "Cherry_Powdery_Mildew",
    "Corn_Common_Rust", "Corn_Healthy", "Corn_Northern_Leaf_Blight",
    "Grape_Black_Rot", "Grape_Esca", "Grape_Healthy", "Grape_Leaf_Blight",
    "Peach_Bacterial_Spot", "Peach_Healthy", "Pepper_Bacterial_Spot", "Pepper_Healthy",
    "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
    "Strawberry_Healthy", "Strawberry_Leaf_Scorch",
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Healthy",
    "Tomato_Late_Blight", "Tomato_Leaf_Mold", "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites", "Tomato_Yellow_Leaf_Curl_Virus"
]

# =============================
# ✅ Streamlit UI
# =============================
st.title("🌿 Plant Disease Detection App")
st.write("Upload a **plant leaf image** to detect its disease using the trained model.")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    try:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction) * 100)
        predicted_disease = class_labels[predicted_class]

        st.success(f"✅ Predicted Class: **{predicted_disease}**")
        st.info(f"🎯 Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
