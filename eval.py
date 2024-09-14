import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import UnetGenerator

# caculate MSELoss of train data

class PairedDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        self.input_dir = input_dir
        self.label_dir = label_dir

        self.input_images = sorted(os.listdir(input_dir))
        self.label_images = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.input_images)
        
    def __getitem__(self, idx):
        y_offset = (256 - 218) // 2
        x_offset = (256 - 178) // 2

        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        label_image_path = os.path.join(self.label_dir, self.label_images[idx])

        input_image = Image.open(input_image_path).convert('RGB')
        label_image = Image.open(label_image_path).convert('RGB')

        input_image = np.array(input_image).reshape(3,218,178) / 255
        label_image = np.array(label_image).reshape(3,218,178) 
        
        new_input_image = np.zeros((3, 256, 256))
        new_input_image[ :, y_offset:y_offset+218, x_offset:x_offset+178] = input_image

        return torch.tensor(new_input_image).float(), np.array(label_image)

input_dir = 'dataset/train/input'
label_dir = 'dataset/train/label'
device = "cuda:1"
model_path = 'models/0620_x16_70_6.6.pt'

dataset = PairedDataset(input_dir=input_dir, label_dir=label_dir)
train_loader = DataLoader(dataset, batch_size=16, shuffle=False)



generator = UnetGenerator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
criterion = nn.MSELoss()

mse_loss=0.
y_offset = (256 - 218) // 2
x_offset = (256 - 178) // 2

generator.eval()

for i, (input_images, target_images) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
    x = input_images.to(device)
    real = target_images.to(device)
    fake = generator(x)
    extracted_fake = fake[:, :, y_offset:y_offset+218, x_offset:x_offset+178] * 255
    extracted_fake = torch.relu(extracted_fake)
    loss = criterion(extracted_fake, real)
    mse_loss += loss.item()

mse_loss = mse_loss/len(train_loader)
print(mse_loss)