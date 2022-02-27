import tensorflow as tf
from ModelName import ModelName
import streamlit as st


class VGG(ModelName):
    __model_path = "models/cassava (1).h5"
    __input_shape = (224, 224)

    def __init__(self):
        pass

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model(VGG.__model_path)
        return model

    @staticmethod
    def preprocessing(image):
        image = super(VGG, VGG).preprocessing(image)
        image = tf.image.resize(image, VGG.__input_shape)
        return image
