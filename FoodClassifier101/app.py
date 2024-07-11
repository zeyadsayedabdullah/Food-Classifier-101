
import gradio as gr
import torch
import os
from model import create_effnetb3_model
from timeit import default_timer as timer
from typing import Tuple, Dict

with open('class_names.txt',"r") as f :
    class_names=  [food.strip() for food in f.readlines()]

effnetb3, effnetb3_transforms = create_effnetb3_model(num_classes=101)

effnetb3.load_state_dict(torch.load(f='effnetb3.pth',map_location=torch.device('cpu')))


def predict(img):
  start = timer()

  img_tensor = effnetb3_transforms(img).unsqueeze(dim=0)

  effnetb3.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb3(img_tensor), dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)

  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  end = timer()
  pred_time = round(end - start,4)

  return pred_labels_and_probs, pred_time


title = "FoodClassifier101 🍕📸"
description = "An EfficientNetB3-based computer vision model designed to classify food images into 101 distinct categories."
example_list = [["examples/" + example] for example in os.listdir("examples")]
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    title=title,
    description=description,
    examples=example_list 
)

# Launch the app!
demo.launch()
