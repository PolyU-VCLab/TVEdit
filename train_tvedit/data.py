from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from io import BytesIO
import glob
from tqdm import tqdm
import json
import random
import os
        

class TVEditDataset(Dataset):
    def __init__(
        self,
        config_path,
        root : str = None,
        condition_size = (512,512),
        datapath:str=None,
        target_size = (512,512),
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
        specific_task: list = None,
    ):
        self.data=[]
        self.sample_num=0
        self.root=root
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()
        self._load_data(config_path)   
        self.datapath =datapath     
        self.train_size = target_size

    def __len__(self):
        return self.sample_num
    def _load_data(self, config_path) -> None:
        count = 0
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.data.append(obj)
                count += 1
        self.sample_num=count
    def point2disk(self,points,H,W):
        points = torch.round(points) # points [n,2]
        # bs = points.shape[0]
        disk_map = torch.zeros((H, W)).long().to(points.device)

        if len(points) == 0:
            return disk_map.unsqueeze(0)

        # Assign values to disk map
        idx = torch.arange(len(points)).to(points.device) + 1
        disk_map[points[:, 1].long(), points[:, 0].long()] = idx
        disk_map = disk_map.unsqueeze(0)
        return disk_map
    def __getitem__(self, idx):
        entry = self.data[idx]

        src_image_path = os.path.join(self.datapath,entry['image1'])
        src_point = np.array(entry['points1'])
        
        tgt_image_path = os.path.join(self.datapath,entry['image2'])
        tgt_point  = np.array(entry['points2'])
        num_points_to_select = np.random.randint(1, min(6, len(src_point) + 1))
        num_points = min(10,len(src_point))
        src_point = src_point[:num_points]
        tgt_point = tgt_point[:num_points]
        src_point[:,0]*=(self.train_size/512) # center crop only
        src_point[:,1]*=(self.train_size/512)
        tgt_point[:,0]*=(self.train_size/512)
        tgt_point[:,1]*=(self.train_size/512)
        mask = (tgt_point[:, 0] > 511) | (tgt_point[:, 1] > 511) | (src_point[:,0] > 511) | (src_point[:,1]>511)
        indices = np.where(~mask)[0]
        src_point = src_point[indices]
        tgt_point = tgt_point[indices]
        rand_swap = random.randint(0,1)
        if rand_swap:
            src_image_path,tgt_image_path = tgt_image_path,src_image_path
            src_point, tgt_point = tgt_point, src_point
        instruction = 'Move the object according to the reference instructions'
        if entry.get('instruct1') is not None:
            if rand_swap:
                instruction= entry['instruct2']
            else:
                instruction= entry['instruct1']
        image = Image.open(src_image_path+'.jpg')
        edited_image = Image.open(tgt_image_path+'.jpg')

        image = image.resize((self.train_size,self.train_size), Image.Resampling.LANCZOS)
        edited_image = edited_image.resize((self.train_size,self.train_size), Image.Resampling.LANCZOS)

        points1 = torch.tensor(src_point) # n,2
        points2 = torch.tensor(tgt_point) # n,2
        disk_map1 = self.point2disk(points1,self.train_size,self.train_size)
        disk_map2 = self.point2disk(points2,self.train_size,self.train_size) # h,w
        

        if random.random() < self.drop_text_prob:
            instruction = " "
        return {
            "image": self.to_tensor(edited_image),
            "condition": self.to_tensor(image),
            "points":(disk_map1,disk_map2),
            "condition_type": "edit", 
            "description": instruction,
            "position_delta": np.array([0, 0]),
            **({"pil_image": [edited_image, combined_image]} if self.return_pil_image else {}),
        }

        