import torch
import torch.nn as nn
from torchvision import models,transforms
from ModelName import ModelName
from efficientnet_pytorch import EfficientNet
import streamlit as st
import numpy as np
import cv2
class EfficientNetB3(ModelName):
    __model_path = "models/224_b3_0.pt"
    __input_shape = (400, 400)


    @st.cache(allow_output_mutation=True)
    def Net(model_name = 'b3', output = 5):
        model = EfficientNet.from_pretrained(f'efficientnet-{model_name}')
        model._fc = nn.Linear(in_features = model._fc.in_features, out_features = output, bias = True)
        return model

    @staticmethod
    def load_model():

        model=EfficientNetB3.Net()
        model.load_state_dict(torch.load(EfficientNetB3.__model_path, map_location=torch.device('cpu'))["model_state_dict"])
        model.eval()

        return model

    @staticmethod
    def preprocessing(image):
        jpg_as_np = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        image=torch.from_numpy(image).to(torch.float)
        image=image.permute(2,1,0)
        transform=transforms.Compose([transforms.Resize(EfficientNetB3.__input_shape)])
        image=transform(image)

        return image

    @staticmethod
    def predict(image,model):
        image=image.unsqueeze(0)
        pred=model(image)
        print(pred)
        pred = nn.functional.softmax(pred, dim=1).data.cpu().numpy().argmax()
        return pred
