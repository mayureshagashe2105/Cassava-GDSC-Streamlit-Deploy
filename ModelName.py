from abc import ABC, abstractmethod
import tensorflow as tf
import cv2

class ModelName(ABC):
     
    __model_path = None
    __input_shape = None
     
    def __init__(self):
       pass

    @staticmethod
    @abstractmethod
    def load_model():
        pass

    @staticmethod
    @abstractmethod
    def preprocessing(image):
        image = tf.io.decode_image(image)
        image = tf.expand_dims(image, 0)
        return image
