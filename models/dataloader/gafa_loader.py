import os
import pickle
from collections import defaultdict
import cv2
import time
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, ImageOps
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class GazeSeqDataset(Dataset):
    def __init__(self, video_path, n_frames=7):
        self.video_path = video_path
        self.n_frames = n_frames

        # load annotation
        with open(os.path.join(video_path, 'annotation.pickle'), "rb") as f:
            anno_data = pickle.load(f)
            #anno_data['index']

        #print(anno_data['index'][0])
        self.bodys = anno_data["bodys"]
        self.heads = anno_data["heads"]
        self.gazes = anno_data["gazes"]
        self.img_index = anno_data['index']

        # abort if no data
        if len(self.gazes) < 1:
            self.valid_index = []
            return

        # extract successive frames
        self.valid_index = []
        for i in range(0, len(self.img_index) - self.n_frames):
            # In condition of GoTK, you can use the following line
            # if self.img_index[i] == self.img_index[i] and i < len(self.gazes):
            # In GAFA dataset, the frame indices are not continuous, so we use the following line.
            if self.img_index[i] + self.n_frames - 1 == self.img_index[i + self.n_frames - 1] and i < len(self.gazes):
                self.valid_index.append(i)
        self.valid_index = np.array(self.valid_index)

        # Head boundig box changed to relative to chest
        self.head_bb = np.vstack(anno_data['head_bb']).astype(np.float32)
        self.body_bb = np.vstack(anno_data['body_bb']).astype(np.float32)
        self.height = self.body_bb[:, 3]
        self.head_bb[:, 0] -= self.body_bb[:, 0]
        self.head_bb[:, 1] -= self.body_bb[:, 1]
        self.head_bb[:, [0, 2]] /= self.body_bb[:, 2][:, None]
        self.head_bb[:, [1, 3]] /= self.body_bb[:, 3][:, None]

        # image transform for body image
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_index)

    def transform(self, item_allframe):
        image = torch.stack(item_allframe['image'])
        head_dir = np.stack(item_allframe['head_dir']).copy()
        body_dir = np.stack(item_allframe['body_dir']).copy()
        gaze_dir = np.stack(item_allframe['gaze_dir']).copy()
        head_bb = np.stack(item_allframe['head_bb']).copy()

        # create mask of head bounding box
        head_mask = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3])
        head_bb_int = head_bb.copy()
        head_bb_int[:, [0, 2]] *= image.shape[3]
        head_bb_int[:, [1, 3]] *= image.shape[2]
        head_bb_int[:, 2] += head_bb_int[:, 0]
        head_bb_int[:, 3] += head_bb_int[:, 1]
        head_bb_int = head_bb_int.astype(np.int64)
        head_bb_int[head_bb_int < 0] = 0
        for i_f in range(head_mask.shape[0]):
            head_mask[i_f, :, head_bb_int[i_f, 1]:head_bb_int[i_f, 3], head_bb_int[i_f, 0]:head_bb_int[i_f, 2]] = 1


        ret_item = {
            'image': image,
            'head_dir': torch.from_numpy(head_dir),
            'body_dir': torch.from_numpy(body_dir),
            'gaze_dir': torch.from_numpy(gaze_dir),
            'head_mask': head_mask,
        }

        return ret_item



    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} >= len {len(self)}")

        idx = self.valid_index[idx]

        item_allframe = defaultdict(list)
        for j in range(idx, idx + self.n_frames):

            # load image
            img_path = os.path.join(self.video_path, f"{self.img_index[j]:06}.jpg")
            img = Image.open(img_path)
            img_ = transform(img)

            item = {
                "image":img_,
                "head_dir": self.heads[j],
                "body_dir": self.bodys[j],
                "gaze_dir": self.gazes[j],
                "head_bb": self.head_bb[j],
            }
            for k, v in item.items():
                item_allframe[k].append(v)
        
        item_allframe = self.transform(item_allframe)
        return item_allframe

def create_gafa_dataset(exp_names, root_dir='./data/preprocessed', n_frames=7):
    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]

    dset_list = []
    for ed in exp_dirs:
        cameras = sorted(os.listdir(ed))
        for cm in cameras:
            if not os.path.exists(os.path.join(ed, cm, 'annotation.pickle')):
                print(f"annotation.pickle not found in {os.path.join(ed, cm)}")
                continue

            dset = GazeSeqDataset(os.path.join(ed, cm), n_frames=n_frames)

            if len(dset) == 0:
                continue
            dset_list.append(dset)

    print("in create_gafa_dataset")

    return ConcatDataset(dset_list)