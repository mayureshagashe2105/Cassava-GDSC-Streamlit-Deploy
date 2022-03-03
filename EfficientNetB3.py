import torch
from torchvision import models,transforms
from ModelName import ModelName
import streamlit as st


class EfficientNetB3(ModelName):
    __model_path = "models/224_b3_0.pt"
    __input_shape = (400, 400)

    def __init__(self):
        pass

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = torch.load(EfficientNetB3.__model_path)
        return model

    @staticmethod
    def preprocessing(image):
        image = super(EfficientNetB3, EfficientNetB3).preprocessing(image)
        image=transforms.Compose([transforms.Resize(EfficientNetB3.__input_shape),transforms.CenterCrop(224)])
        image=image.float()
        return image







