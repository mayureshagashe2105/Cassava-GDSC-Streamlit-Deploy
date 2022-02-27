import tensorflow as tf

class vgg:
    def __init__(self):
        pass
    @staticmethod
    def load_model():
        model = tf.keras.models.load_model(r"C:\Users\sahas\OneDrive\Documents\visual studio code\cassava\cassava (1).h5")
        return model

    @staticmethod
    def preprocess(image):
        image = tf.io.decode_image(image)
        image = tf.expand_dims(image,0)
        image = tf.image.resize(image,(224,224))
        return image

