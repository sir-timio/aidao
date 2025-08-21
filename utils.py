import io
import os

from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset


PUBLIC_DATA_FOLDER_PATH = 'images'
PUBLIC_DATA_DESCRIPTION_PATH = 'images/public_description.csv'
PRIVATE_DATA_FOLDER_PATH = 'private_test'
PRIVATE_DATA_DESCRIPTION_PATH = 'private_description.csv'


def show_photos(photos):
    buffer = []
    for photo in photos:
        buffer.append(
            Image.open(io.BytesIO(photo)).resize((300, 200), resample=Image.BILINEAR)
        )
    
    widths, heights = zip(*(img.size for img in buffer))
    
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in buffer:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im

    
def train_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    images_cnt = 0
    
    for batch in tqdm(train_loader, total=len(train_loader)):
        images = batch['photo'].to(device)
        labels = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images).squeeze()
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        train_loss += loss.detach().cpu().item()
        images_cnt += images.shape[0]
        train_acc += (labels == (outputs >= 0.6)).sum().cpu().item()
        
        optimizer.step()
    return train_loss / images_cnt, train_acc / images_cnt


def test_epoch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    images_cnt = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            images = batch['photo'].to(device)
            labels = batch['target'].to(device)
            
            outputs = model(images).squeeze()

            loss = criterion(outputs, labels.float())
            test_loss += loss.detach().cpu().item()
            images_cnt += images.shape[0]
            test_acc += (labels == (outputs >= 0.6)).sum().cpu().item()
    return test_loss / images_cnt, test_acc / images_cnt


def plot_history(train_loss, test_loss, train_acc, test_acc):
    plt.figure(figsize=(15, 9))
    plt.subplot(1, 2, 1)
    x = range(1, len(train_loss) + 1)
    plt.plot(x, train_loss, '-bo', label='train loss')
    plt.plot(x, test_loss, '-ro', label='test loss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    
    plt.subplot(1, 2, 2)
    x = range(1, len(train_acc) + 1)
    plt.plot(x, train_acc, '-bo', label='train acc')
    plt.plot(x, test_acc, '-ro', label='test acc')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    
    plt.show()
    
    
def print_model_params_required_grad(model):
    left = 0
    left_state = None
    for right, param in enumerate(model.parameters()):
        if left_state is None:
            left_state = param.requires_grad
        right_state = param.requires_grad
        if right_state == left_state:
            continue
        print(f'[{left} - {right - 1}]: {left_state} × {right - left}')
        left = right
        left_state = right_state
    right += 1
    print(f'[{left} - {right - 1}]: {left_state} × {right - left}')
    
    
def get_predictions(model, device, val_loader, add_sigmoid=False):
    model.eval()
    
    y_real = []
    y_pred = []
    pass_ids = []
    plan_sides = []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader)):
            images = batch['photo'].to(device)
            
            if add_sigmoid:
                outputs = torch.sigmoid(model(images).squeeze())
            else:
                outputs = model(images).squeeze()
            
            y_pred.extend(outputs.detach().cpu().numpy())
            
            pass_ids.extend(batch['pass_id'])
            plan_sides.extend(batch['plan_side'])
            
    return pd.DataFrame.from_dict({
        'pass_id': pass_ids,
        'prediction': y_pred,
        'plan_side': plan_sides,
    })


class PhotoDataset(Dataset):
    def __init__(self, img_dir_path, target_map, data_description, img_preprocess = lambda x: x):
        self.img_dir = img_dir_path
        self.target_map = target_map
        self.targets = sorted(list(self.target_map.keys()))
        self.img_preprocess = img_preprocess
        self.data_description = data_description
        
    def __len__(self):
        return len(self.target_map)
    
    def __getitem__(self, idx):
        key = self.targets[idx]
        key_description = self.data_description.loc[key]
        with open(os.path.join(self.img_dir, key), 'rb') as f:
            img = self.img_preprocess(f.read())
        return {
            'photo': img,
            'target': self.target_map[key],
            'pass_id': key_description.pass_id,
            'plan_side': key_description.plan_side,
        }


def create_dataloader(
    img_dir_path: str,
    target_map,
    description,
    batch_size: int,
    preprocessor,
    num_load_workers: int,
):

    dataset = PhotoDataset(
        img_dir_path=img_dir_path,
        target_map=target_map,
        data_description=description,
        img_preprocess=preprocessor,
    )

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_load_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )