import requests
from PIL import Image
import json
import base64
from io import BytesIO

print("========== Temporary client server ==========")
img_path = './dataset/detection/img_00001.jpg'
img = Image.open(img_path, mode='r')
img = img.convert('RGB')
buffered = BytesIO()
img.save(buffered, format="jpeg")
base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
files = {
    "image": base64_string,
}

r = requests.post("http://127.0.0.1:8000/detect", json=files)
print(r)