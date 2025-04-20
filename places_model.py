from collections import OrderedDict
import torch
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

# Load categories


def load_categories():
    file_name = 'categories_places365.txt'
    if not os.path.exists(file_name):
        url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        urllib.request.urlretrieve(url, file_name)
    classes = [line.strip().split(' ')[0][3:] for line in open(file_name)]
    return classes

# Load model


def load_model():
    model = models.resnet18(num_classes=365)
    model_file = 'resnet18_places365.pth.tar'
    if not os.path.exists(model_file):
        url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
        urllib.request.urlretrieve(url, model_file)
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

    # Remove 'module.' prefix if it exists
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    return model

# Predict scene


def predict_scene(image_path, model, classes):
    center_crop = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    input_img = center_crop(img).unsqueeze(0)
    with torch.no_grad():
        logit = model(input_img)
        probs = torch.nn.functional.softmax(logit, 1)
        top5 = probs.topk(5)
    predictions = [(classes[idx], prob.item())
                   for idx, prob in zip(top5.indices[0], top5.values[0])]
    return predictions
