import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
from vgg import vgg

st.title("Cassava Disease Detection")

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
'''if image_file is not None:
    print(image_file.getvalue())'''

option = st.selectbox(
     'Choose the Model ',
     ('VGG16', 'EfficientNet', 'MobileNet'))
     

st.write('You selected:', option)

if option=='VGG16':
    model = vgg.load_model()
    if image_file is not None:
        image = vgg.preprocess(image_file.getvalue())

        prediction = model.predict(image)
        print(prediction)
elif option=='EfficientNet':
    pass
else:
    pass
