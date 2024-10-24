import torch._utils_internal
import os
import cv2
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch
from glob import glob
import numpy as np
from django.conf import settings

# 가상 환경은 my_conda로 할 것

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.models import detection
from tqdm import tqdm

############# module import ##############

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, BatchSampler
from PIL import Image
import torch
import torchtext
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import random
from matplotlib.pyplot import imshow

import nltk, re, string, collections
from nltk.util import ngrams
import collections
#image file is truncated -> 오류해결코드
#image 파일이 잘려있으면 나오는 오류
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import OrderedDict


# IMG_DIR = './data/j_images'
# img_list3 = glob("./data/j_images/*/*/*.jpg")
# img_list4 = glob("./data/j_images/*/*/*.jpeg")
# img_list = img_list3 + img_list4


# img_list = glob("./shoe_board/_media/*/*") #django에서 input받은 upload이미지 경로
detect_model = YOLO("/home/psw1022s/dataset/runs/detect/train4/weights/best.pt") 

import io
from PIL import Image

def save_cropped_region(image, box, output_path):
    dir_name = os.path.dirname(output_path)
    os.makedirs(dir_name, exist_ok=True)
    
    x1, y1, x2, y2 = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

def image_crop(_image, filename, confidence_threshold=0.6):
  if isinstance(_image, str):
    origin_image = cv2.imread(_image)
  else:
    origin_image = Image.open(io.BytesIO(_image))
    numpy_image = np.array(origin_image)  
    origin_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
  print('*'* 10)
  results = detect_model.predict(source=origin_image)
  print('-'* 10)
  
  _filename = filename.split('.')
  filename, ext = ".".join(_filename[:-1]), _filename[-1]
  
  d = origin_image.copy()
  cropped_count = 0
  return_results = {}
  
  for result in results:
    boxes = result.boxes.xyxy
    confs = result.boxes.conf
    clss = result.boxes.cls
    
    for i in range(len(boxes)):
      conf = confs[i]
      x1, y1, x2, y2 = map(int, boxes[i])
      d = cv2.rectangle(d, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cls2 = clss[i]
      
      if conf > confidence_threshold:
        output_path = f"{filename}/cropped/{cropped_count}.{ext}"
        output_path = os.path.join(settings.MEDIA_ROOT, output_path)
        return_results[output_path] = conf
        
        save_cropped_region(origin_image, {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}, output_path)
        cropped_count+=1
        # print(f"Saved cropped region with {conf:.2f} confidence for {img_file} as {output_path}")
      
  return return_results

import glob
import pickle
# with open('text_vector.pkl', 'wb') as f:
#     pickle.dump(text_vector, f)
# model = torch.load('best_model_33.pt')

with open('/home/psw1022s/jh/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
    
with open('/home/psw1022s/jh/text_vector.pkl', 'rb') as f:
    text_vector = pickle.load(f).float()
indices_list = []

def clip_shoes(output_path : dict):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

  test_model = torch.load("/home/psw1022s/jh/best_model_33.pt")
  model.load_state_dict(test_model, strict=False)
      
  # output_path = glob('/home/psw1022s/shoe_board/_media/*/*.jpg')
  indices_list = []
  sort_dict = {}
  
  for output in output_path.keys():
    print(output)
  
    img = preprocess(Image.open(output).resize((640, 640))).to(device) #type:ignore

    image_features = model.encode_image(img.unsqueeze(0))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.detach().cpu().float()

    similarity = (100.0 * image_features @ text_vector.T).type(torch.float32)
    values, indices = similarity[0].topk(10)
    
    for v,i in zip(values, indices):
      sort_dict[i] = v
  
    # indices_list = [corpus[result] for result in indices]
    # for result in indices:
    #   indices_list.append(corpus[result]) 
  sort_dict = dict(sorted(sort_dict.items(), key=lambda x:x[1], reverse = True))
  # return indices_list[::2]
  a = sort_dict.keys()
  # print(corpus[a])
  return [corpus[tmp] for tmp in a][:10]

  