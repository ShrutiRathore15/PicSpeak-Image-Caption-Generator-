import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image
import requests
from io import BytesIO

def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Resize the image to match VGG16 input size
    image = np.array(image)
    image = preprocess_input(image)
    return image

# Load your trained model
model = tf.keras.models.load_model("C:/Users/shrut/PycharmProjects/IC_1.1/best_mode_vgg_40.h5")

# Load the tokenizer
with open('C:/Users/shrut/PycharmProjects/IC_1.1/tokenizer40_new.pickle', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Set custom web page title
st.set_page_config(page_title="Image Caption Generator", page_icon="📷")

with st.sidebar:
    st.title("Every picture has a story; we help you tell it!")
    st.title("Turning Moments into Narratives with our Image Caption Generator")

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        image = Image.open(uploaded_image)
        #st.image(image, caption="Uploaded Image", use_column_width=True)




        # Extract features using VGG16
        new_image = preprocess_input(image)
        vgg_model = VGG16()
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)

        # Max caption length
        max_caption_length = 35


        # Define function to get word from index
        def idx_to_word(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None


        # Generate caption using the model
        def predict_caption(model, image, tokenizer, max_length):
            # add start tag for generation process
            in_text = 'startseq'
            # iterate over the max length of sequence
            for i in range(max_length):
                # encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad the sequence
                sequence = pad_sequences([sequence], max_length)
                # predict next word
                yhat = model.predict([image, sequence], verbose=0)
                # get index with high probability
                yhat = np.argmax(yhat)
                # convert index to word
                word = idx_to_word(yhat, tokenizer)
                # stop if word not found
                if word is None:
                    break
                # append word as input for generating the next word
                in_text += " " + word
                # stop if we reach end tag
                if word == 'endseq':
                    break
            return in_text


        # Generate caption
        generated_caption = predict_caption(model, new_image_features, tokenizer, max_length=35)
        generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
        generated_caption = generated_caption.capitalize()

    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )
