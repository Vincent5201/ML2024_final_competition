import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

from models import UnetGenerator, Discriminator

# train model

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated_image

class PairedDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        self.input_dir = input_dir
        self.label_dir = label_dir

        self.input_images = sorted(os.listdir(input_dir))
        self.label_images = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.input_images) * 16

    def __getitem__(self, idx):
        angle = idx % 16
        image_idx = idx // 16
        y_offset = (256 - 218) // 2
        x_offset = (256 - 178) // 2

        input_image_path = os.path.join(self.input_dir, self.input_images[image_idx])
        label_image_path = os.path.join(self.label_dir, self.label_images[image_idx])
        input_image = Image.open(input_image_path).convert('RGB')
        label_image = Image.open(label_image_path).convert('RGB')
        
        input_image = np.array(input_image).reshape(3,218,178) / 255
        label_image = np.array(label_image).reshape(3,218,178) / 255

        # put 218*178 image in the middle of 256*256
        new_input_image = np.zeros((3, 256, 256))
        new_label_image = np.zeros((3, 256, 256))
        new_input_image[:, y_offset:y_offset+218, x_offset:x_offset+178] = input_image
        new_label_image[:, y_offset:y_offset+218, x_offset:x_offset+178] = label_image
        
        # rotate 90 degree
        while angle > 3:
            angle -= 4
            new_input_image = np.rot90(new_input_image, k=1, axes=(1, 2)).copy()
            new_label_image = np.rot90(new_label_image, k=1, axes=(1, 2)).copy()
        
        # rotate 45 degree
        if angle > 1:
            angle -= 2
            rotated_input_image = np.zeros_like(new_input_image)
            rotated_label_image = np.zeros_like(new_label_image)
            for i in range(3):
                rotated_input_image[i] = rotate_image(new_input_image[i], 45)
                rotated_label_image[i] = rotate_image(new_label_image[i], 45)
            new_input_image = rotated_input_image.copy()
            new_label_image = rotated_label_image.copy()

        # flip
        if angle:
            new_input_image = np.fliplr(new_input_image.transpose(1, 2, 0)).transpose(2, 0, 1).copy()
            new_label_image = np.fliplr(new_label_image.transpose(1, 2, 0)).transpose(2, 0, 1).copy()
        
        return torch.tensor(new_input_image).float(), torch.tensor(new_label_image).float()

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()

    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha* self.l1(fake, real)
        return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss

input_dir = 'dataset/train/input'
label_dir = 'dataset/train/label'
device = "cuda:1"
num_epochs = 100
small_g = 7
batch = 16
lr = 0.0005
alpha = 100

dataset = PairedDataset(input_dir=input_dir, label_dir=label_dir)
train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)

generator = UnetGenerator().to(device)
discriminator = Discriminator().to(device)

g_criterion = GeneratorLoss(alpha=alpha)
d_criterion = DiscriminatorLoss()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

generator.train()
discriminator.train()

for epoch in range(num_epochs):
    ge_loss=0.
    de_loss=0.
    for i, (input_images, target_images) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
        x = input_images.to(device)
        real = target_images.to(device)

        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ge_loss += g_loss.item()
        de_loss += d_loss.item()

    g_loss = ge_loss/len(train_loader)
    d_loss = de_loss/len(train_loader)
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f]" % (epoch+1, num_epochs, g_loss, d_loss))
    if g_loss < small_g:
        torch.save(generator.state_dict(), f"models/generator_epoch_{epoch + 1}.pt")
    small_g = min(small_g, g_loss)