from huggingface_hub import snapshot_download
import os
import requests
import shutil
from subprocess import call


def download_file(url, destination):
    with requests.get(url, stream=True) as r:
        with open(destination, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def download_models():
    print(f"1. [NANOVLM] Downloading models to models/...")
    snapshot_download(repo_id="Efficient-Large-Model/VILA-2.7b", local_dir="models/VILA-2.7b")
    snapshot_download(repo_id="openai/clip-vit-large-patch14-336", local_dir="models/clip-vit-large-patch14-336")

    print(f"2. [FASTERSAM] Downloading models to models/...")
    download_file('https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/visual.onnx', 'models/onnx32-open_clip-ViT-B-32-openai-visual.onnx')
    download_file('https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx', 'models/onnx32-open_clip-ViT-B-32-openai-textual.onnx')
    download_file('https://drive.google.com/uc?export=download&id=10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV', 'models/yolov8s-seg.pt')
    call(['yolo', 'mode=export', 'model=models/yolov8s-seg.pt', 'format=onnx', 'imgsz=512,512'])
