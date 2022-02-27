import tensorflow as tf
from ModelName import ModelName
import streamlit as st


class MobileNetV2(ModelName):
    __model_path = "models/MobileNetV2_for_Cassava.h5"
    __input_shape = (400, 400)

    def __init__(self):
        pass

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model(MobileNetV2.__model_path)
        return model

    @staticmethod
    def preprocessing(image):
        image = super(MobileNetV2, MobileNetV2).preprocessing(image)
        image = tf.image.resize(image, MobileNetV2.__input_shape)
        image = tf.cast(image, 'float32') / 255.0
        image = MobileNetV2.zoom_images(image)
        return image

    @classmethod
    def zoom_images(cls, tensor):
        tensor = tf.image.central_crop(tensor, 0.80)
        return tensor
