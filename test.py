import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

from models import UnetGenerator

# generate output file of test data

class PairedDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_images = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        y_offset = (256 - 218) // 2
        x_offset = (256 - 178) // 2
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_image_path).convert('RGB')
        input_image = np.array(input_image).reshape(3,218,178) / 255
        new_input_image = np.zeros((3, 256, 256))
        new_input_image[ :, y_offset:y_offset+218, x_offset:x_offset+178] = input_image

        return torch.tensor(new_input_image).float()

input_dir = 'dataset/test/input'
device = "cuda:1"
model_path = 'models/0620_x16_70_6.6.pt'

dataset = PairedDataset(input_dir=input_dir)
train_loader = DataLoader(dataset, batch_size=16, shuffle=False)

generator = UnetGenerator().to(device)
generator.load_state_dict(torch.load(model_path))


y_offset = (256 - 218) // 2
x_offset = (256 - 178) // 2
total = []

generator.eval()

for i, input_images in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
    x = input_images.to(device)
    fake = generator(x)
    extracted_fake = fake[:, :, y_offset:y_offset+218, x_offset:x_offset+178] * 255
    extracted_fake = torch.relu(extracted_fake)
    extracted_fake = torch.round(extracted_fake).reshape(extracted_fake.shape[0], -1)
    extracted_fake = extracted_fake.detach().cpu().numpy().astype(int)
    total.extend(extracted_fake)
    
total = np.array(total)
df = pd.DataFrame(total)
df.index = range(total.shape[0])
df.columns = range(total.shape[1])
df.to_csv('out.csv', index_label='index')