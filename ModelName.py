from abc import ABC, abstractmethod
import tensorflow as tf
class ModelName(ABC):
    def __init__(self):
        __model_path = None
        __input_shape = None

    @staticmethod
    @abstractmethod
    def load_model(self):
        pass

    @staticmethod
    @abstractmethod
    def preprocessing(self,image):
        image = tf.io.decode_image(image)
        image = tf.expand_dims(image, 0)
        return image
