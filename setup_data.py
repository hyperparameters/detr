from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import numpy as np

DATA_HOME= "datasets/"

import os
import wget
import subprocess

def download(dataset_name, data_source):
    data_root = os.path.join(DATA_HOME,dataset_name)
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
    for k,v in data_source.items():
        
        file = os.path.basename(v)
        if not os.path.isfile(os.path.join(data_root,file)):
            print(f"downloading {k}")
            wget.download(v,data_root)
            print(f"done downloading {k}")
        else:
            print(f"{file} exist skipping..")
        
        print(f"extracting zip..")
#         os.system(f"unzip -q {os.path.join(data_root,file)}")
        subprocess.run(["unzip","-q",os.path.join(data_root,file),"-d",data_root],capture_output=True)
        print(f"done extracting {file}")
        
data_source = {"annotations":os.environment("annotations"),
              "train": os.environment("train"),
              "val":os.environment("val")}


dataset_name= "crowdhuman"
download(dataset_name,data_source)