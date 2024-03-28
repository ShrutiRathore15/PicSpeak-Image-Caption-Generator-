import streamlit as st
#import pyautogui
import base64
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image
import requests
from io import BytesIO


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }

    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background(r'wallpaper.jpg')


# Set custom web page title
# st.set_page_config(page_title="Image Caption Generator", page_icon="📷")


def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Resize the image to match VGG16 input size
    image = np.array(image)
    image = preprocess_input(image)
    return image


# Load MobileNetV2 model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model("best_mode_vgg_40.h5")

# Load the tokenizer
with open('tokenizer40_new.pickle', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with st.sidebar:
    st.title("Every picture has a story; we help you tell it!")
    st.title("Turning Moments into Narratives with our Image Caption Generator")

# Streamlit app
st.title("PicSpeak: Visual Narratives Unleashed")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key='image')

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        # image = load_img(uploaded_image, target_size=(224, 224))
        # image = img_to_array(image)
        # image = image.reshape((-1, 224, 224, 3))
        # image = preprocess_input(image)

        image = Image.open(uploaded_image)
        image = preprocess_image(uploaded_image)
        # image = preprocess_input(image)

        # Extract features using VGG16
        image_features = vgg_model.predict(np.array([image]), verbose=0)

        # Max caption length
        max_caption_length = 35


        # Define function to get word from index
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
            )


        # Generate caption using the model
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption


        # Generate caption
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Remove startseq and endseq
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('##')
#if st.button("Reset"):
   # pyautogui.hotkey("ctrl", "F5")
