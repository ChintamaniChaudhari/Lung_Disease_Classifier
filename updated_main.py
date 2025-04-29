import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import keras
import tensorflow_hub as hub

from updated_util import classify, set_background

def show_output_chart(predictions, class_names):
    colors = ['#fddccf', '#fce4ec', '#8e44ad', '#d6eaf8', '#fddccf']  # customize as needed

    fig = go.Figure()

    for i, (score, label) in enumerate(zip(predictions, class_names)):
        fig.add_trace(go.Bar(
            x=[score * 100],
            y=[label],
            orientation='h',
            marker=dict(color=colors[i % len(colors)]),
            text=f'{int(score * 100)}%',
            textposition='inside',
            insidetextanchor='start',
            hoverinfo='none'
        ))

    fig.update_layout(
        title='Output',
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(autorange='reversed'),
        height=300,
        margin=dict(l=100, r=20, t=40, b=20),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)


set_background("BackGround.jpg")

with st.container():
    st.markdown("<div class='block-container'>", unsafe_allow_html=True)

# set title
st.title('Multiple Lung Disease classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
#model = load_model("lung_disease_mobilenetv2.h5")
# Example: if you used hub.KerasLayer
model = load_model("lung_disease_mobilenetv2.h5", custom_objects={'KerasLayer': hub.KerasLayer})

                   
# load class names
with open("labels.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score, all_preds = classify(image, model, class_names)

    st.subheader("Output")

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    class_name, conf_score, all_preds = classify(image, model, class_names)

    # show predicted label
    st.write("### Predicted: `{}` with {:.2f}% confidence".format(class_name, conf_score * 100))

    # display horizontal bar chart like your image
    
    for i, score in enumerate(all_preds):
        st.markdown(f"**{class_names[i]}**")
        st.progress(float(score))
    st.markdown("</div>", unsafe_allow_html=True)
