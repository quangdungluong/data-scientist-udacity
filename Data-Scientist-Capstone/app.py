"""Flask app"""
import os

import orjson
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, send_from_directory
from PIL import Image

from src.model import ChestXRayModel

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
UPLOAD_FOLDER = 'uploads'

# Create model
CONFIG_PATH = "./config/train_config.json"
params = orjson.loads(open(CONFIG_PATH, "rb").read())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChestXRayModel(num_classes=params['num_classes'])
model.eval()


def process_image(image_path):
    """
    The function `process_image` takes an image path as input, applies a series of transformations to
    the image, and returns the processed image.
    
    :param image_path: The path to the image file that you want to process
    :return: a processed image tensor.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])

    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = image[None, :].to(device)
    return image

def predict(model, image):
    """
    The function "predict" takes a model and an image as input, and returns the predicted class label
    and the probabilities of each class.
    
    :param model: The "model" parameter refers to a trained machine learning model that is capable of
    making predictions on images. It could be a neural network, a deep learning model, or any other
    model that has been trained to classify or analyze images
    :param image: The "image" parameter is the input image that you want to make predictions on. It
    should be in a format that is compatible with the model you are using
    :return: two values: the prediction and the probabilities.
    """
    output = model(image)
    probs = torch.nn.Softmax(dim=1)(output).detach().numpy()[0]
    _, prediction = output.max(1)
    return prediction, probs

def allowed_file(filename):
    """
    The function checks if a given filename has an allowed file extension.
    
    :param filename: The filename parameter is a string that represents the name of a file
    :return: a boolean value indicating whether the file extension of the given filename is in the list
    of allowed extensions.
    """
    """Check allow file"""
    return filename.split('.')[-1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    """Home"""
    return render_template('index.html', label='Hiiii', imagesource='./static/img/b.png', returnJson={})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Upload and process file"""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = process_image(image_path=file_path)
            output, probs = predict(model, image)
        ## Get Ground truth label
        return_json = {}
        for i, prob in enumerate(probs):
            return_json[params['labels_map'][i]] = round(prob*100,3)
        return_json = dict(sorted(return_json.items(), key=lambda item: item[1], reverse=True))
    return render_template('index.html', label=params['labels_map'][output], imagesource=file_path, returnJson=return_json)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Save uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs("./uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)