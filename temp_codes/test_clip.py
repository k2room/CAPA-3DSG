import json
import open3d as o3d
import torch
import torch.nn.functional as F
from collections.abc import Iterable
import copy
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
import pickle
import gzip
from tqdm import tqdm
from pathlib import Path

root_path = "/home/main/workspace/k2room2/gpuserver00_storage/CAPA/FunGraph3D"
root_path = "/home/main/workspace/k2room2/gpuserver00_storage/CAPA/SceneFun3D_Graph"

all_labels_embeddings = np.load(root_path+'/all_labels_clip_embeddings.npy')
with open(root_path+'/all_labels.json', 'r') as f:
    all_labels = json.load(f)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

while True:
    text_input = input("Enter text query (or 'exit' to quit): ")
    if text_input.lower() == 'exit':
        break

    inputs = processor(text=[text_input], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, p=2, dim=-1).cpu().numpy()

    similarities = np.dot(all_labels_embeddings, text_features.T).squeeze(1)
    topk_indices = np.argsort(-similarities)[:5]

    print("Top 5 similar labels:")
    for idx in topk_indices:
        print(f"\tLabel: {all_labels[idx]},     Similarity: {similarities[idx]:.4f}")