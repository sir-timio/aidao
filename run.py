import copy
import csv
from enum import Enum
import io
import json
import os
import typing as t
import io
from tqdm import tqdm

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
from torch.nn import functional as F
import torch.utils.data as td
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

#############################################
############### DATASET #####################
#############################################

PUBLIC_DATA_FOLDER_PATH = '...'
PUBLIC_DATA_DESCRIPTION_PATH = 'public_description.csv'
PRIVATE_DATA_FOLDER_PATH = 'private_test'
PRIVATE_DATA_DESCRIPTION_PATH = 'private_description.csv'

BATCH_SIZE = 64
NUM_WORKERS = 0
MODEL_PT_NAME = "model.pt"

MODEL_ID = "efficientformer_l1.snap_dist_in1k"

data_config = timm.data.resolve_data_config(timm.get_pretrained_cfg(MODEL_ID).__dict__)
TRANSFORMS = timm.data.create_transform(**data_config, is_training=False)


class CarSide(Enum):
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    UNKNOWN = 4


fact_side_mapping = {
    'front': CarSide.FRONT.value,
    'back': CarSide.BACK.value,
    'left': CarSide.LEFT.value,
    'right': CarSide.RIGHT.value,
    'unknown': CarSide.UNKNOWN.value
}

class TestDataset(Dataset):
    def __init__(self, img_dir, description, transform=None):
        self.img_dir = img_dir
        self.description = description.reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.description)

    def __getitem__(self, idx):
        row = self.description.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        data = {
            'image': image,
            'pass_id': row['pass_id'],
            'plan_side': row['plan_side']
        }
        return data

description = pd.read_csv(PRIVATE_DATA_DESCRIPTION_PATH, index_col='filename').sort_index()
test_dataset = TestDataset(img_dir=PRIVATE_DATA_FOLDER_PATH, description=description, transform=TRANSFORMS)

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#############################################
############### MODEL #######################
#############################################

device = torch.device('cpu')
model = torch.jit.load(MODEL_PT_NAME, map_location=device)
model.to(device)
model.eval()

#############################################
############### predictions #################
#############################################

final_predictions = []
pass_ids = []
plan_sides = []
predicted_fact_sides = []


def get_predictions(model, dataloader, device, fact_side_mapping=fact_side_mapping, side_confidence_threshold=0.7):
    model.eval()
    model.to(device)
    
    final_predictions = []
    pass_ids = []
    plan_sides = []
    predicted_fact_sides = []
    fact_side_confidences = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            batch_pass_ids = batch['pass_id']
            batch_plan_sides = batch['plan_side']
            
            _, _, fact_side_logits, final_logits = model(images)
            
            final_probs = torch.sigmoid(final_logits.squeeze())
            
            fact_side_probs = torch.softmax(fact_side_logits, dim=1)
            max_probs, fact_side_preds = torch.max(fact_side_probs, dim=1)
            fact_side_confidences.extend(max_probs.cpu().numpy())
            predicted_fact_sides.extend(fact_side_preds.cpu().numpy())
            
            final_predictions.extend(final_probs.cpu().numpy())
            pass_ids.extend(batch_pass_ids)
            plan_sides.extend(batch_plan_sides)
    
    inverse_fact_side_mapping = {v: k for k, v in fact_side_mapping.items()}
    predicted_fact_sides_names = [inverse_fact_side_mapping[label] for label in predicted_fact_sides]
    
    predictions_df = pd.DataFrame({
        'pass_id': pass_ids,
        'prediction': final_predictions,
        'plan_side': plan_sides,
        'predicted_fact_side': predicted_fact_sides_names,
        'fact_side_confidence': fact_side_confidences
    })
    
    predictions_df['prediction'] = predictions_df.apply(
        lambda row: 1.0 if (row['plan_side'] != row['predicted_fact_side'] and row['fact_side_confidence'] >= side_confidence_threshold) else row['prediction'],
        axis=1
    )
    
    return predictions_df


solution = get_predictions(model=model, dataloader=test_loader, device=device)
print("INFERENCE DONE")
solution = solution[['pass_id', 'prediction']].groupby('pass_id').max()
solution.to_csv('./predictions.csv')
print("SAVED")

