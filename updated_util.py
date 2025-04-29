
import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        [data-testid="stAppViewContainer"] > .main::before {{
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 0;
        }}

        .block-container {{
            position: relative;
            z-index: 1;
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            color: white;
        }}

        h1, h2, h3, h4, h5, h6, p, label, .stButton > button {{
            color: white !important;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    data = np.expand_dims(image_array, axis=0)
    prediction = model.predict(data)[0]
    top_index = np.argmax(prediction)
    class_name = class_names[top_index]
    confidence_score = prediction[top_index]
    return class_name, confidence_score, prediction
