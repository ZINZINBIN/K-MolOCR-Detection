# Flask package
from flask import Flask, request, render_template
from flask_restx import Api, Resource

# parsing
import argparse

# Deep learning framework
import torch
from src.models.SSD300.model import SSD300
from src.detect import detect

# Image preprocessing package
from PIL import Image
import json
from io import BytesIO
import base64

print("========== Start detection server ==========")

# torch device state
print("=============== device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

def parsing():
    parser = argparse.ArgumentParser(description="Start detection server for K-MolOCR")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SSD")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.2)
    parser.add_argument("--max_overlap", type = float, default = 0.5)
    parser.add_argument("--top_k", type = int, default = 5)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    args = vars(parser.parse_args())
    return args

# parsing
args = parsing()
tag = args['tag']
    
# device allocation
if(torch.cuda.device_count() >= 1):
    device = "cuda:" + str(args["gpu_num"])
else:
    device = 'cpu'
    
save_best_dir = "./weights/{}_best.pt".format(tag)

# load 
print("=========== Load detection model ===========")
model = SSD300(4)
model.to(device)
model.eval()    
model.load_state_dict(torch.load(save_best_dir, map_location = device))

print("=============== Flask API on ===============")
app = Flask(__name__)
# api = Api(app)
    
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/detect", methods = ["POST"])
def detect_mol():
    
    if request.method == 'POST':
        
        json_data = request.get_json()
        dict_data = json.dumps(json_data)
        dict_data = json.loads(dict_data)
        file = dict_data['image']
        
        if not file:
            return render_template("index.html", label = 'No files')

        try:
            img = base64.b64decode(file)
            img = BytesIO(img)
            img = Image.open(img)
            
        except:
            print("File upload error ")
            return render_template("index.html", label = 'File upload error')

        try:
            annot, is_success = detect(img, model, device, min_score = args['min_score'], max_overlap = args['max_overlap'], top_k = args['top_k'])
            annot.save("./results/request_detection.jpg")
            print("Detection process complete")
 
            if is_success:
                return render_template("index.html", label = 'Detection complete')
            else:
                return render_template("index.html", label = 'Detection failed')
            
        except:
            return render_template("index.html", label = 'Detection process error')
        
if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 8000)