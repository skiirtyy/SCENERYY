from collections import OrderedDict
from flask import Flask, request, render_template_string
import torch
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

app = Flask(__name__)

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

    # Remove 'module.' prefix if present
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

# Homepage with upload form


@app.route('/')
def home():
    return render_template_string("""
    <!doctype html>
    <title>Scene Recognition</title>
    <h1>Upload an image to predict scene</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*">
      <input type="submit" value="Predict">
    </form>
    """)

# Prediction route


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image file provided', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    img_path = 'temp.jpg'
    file.save(img_path)

    classes = load_categories()
    model = load_model()
    predictions = predict_scene(img_path, model, classes)

    result_html = "<h2>Top Predictions:</h2><ul>"
    for scene, prob in predictions:
        result_html += f"<li>{scene} ({prob:.4f})</li>"
    result_html += "</ul>"

    return render_template_string("""
    <!doctype html>
    <title>Result</title>
    <h1>Prediction Result</h1>
    """ + result_html + """
    <a href="/">Try another image</a>
    """)

    
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)