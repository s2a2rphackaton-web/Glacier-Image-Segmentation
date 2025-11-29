# pip install torch torchvision numpy pillow tifffile

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader


class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        self.Ws = nn.Sequential(nn.Conv2d(s_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        self.psi = nn.Sequential(nn.Conv2d(out_channels, 1, kernel_size=1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, s):
        g1 = self.Wg(g); s1 = self.Ws(s); out = self.relu(g1 + s1)
        psi = self.psi(out); return s * psi

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class LightAttentionUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)
    
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(g_channels=256, s_channels=256, out_channels=128)
        self.dec4 = conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(g_channels=128, s_channels=128, out_channels=64)
        self.dec3 = conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(g_channels=64, s_channels=64, out_channels=32)
        self.dec2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(g_channels=32, s_channels=32, out_channels=16)
        self.dec1 = conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))
        d4 = self.upconv4(bottleneck); s4 = self.ag4(g=d4, s=e4); d4 = torch.cat((d4, s4), dim=1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); s3 = self.ag3(g=d3, s=e3); d3 = torch.cat((d3, s3), dim=1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); s2 = self.ag2(g=d2, s=e2); d2 = torch.cat((d2, s2), dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); s1 = self.ag1(g=d1, s=e1); d1 = torch.cat((d1, s1), dim=1); d1 = self.dec1(d1)
        return torch.sigmoid(self.final_conv(d1))


class GlacierTestDataset(Dataset):
    def __init__(self, imagepath_dict, bands=['Band1', 'Band2', 'Band3', 'Band4', 'Band5']):
        self.imagepath_dict = imagepath_dict
        self.bands = sorted(bands)
        ref_band_folder = self.imagepath_dict[self.bands[0]]
        self.filenames = sorted([f for f in os.listdir(ref_band_folder) if f.endswith('.tif')])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_bands = []
        for band in self.bands:
            band_folder = self.imagepath_dict[band]
            file_path = os.path.join(band_folder, filename)
            band_arr = np.array(Image.open(file_path), dtype=np.float32)
            if band_arr.ndim == 3: band_arr = band_arr[..., 0]
            input_bands.append(band_arr)
        input_tensor = np.stack(input_bands, axis=0)
        tile_id = os.path.splitext(filename)[0].replace('img', '')
        return torch.from_numpy(input_tensor), tile_id

def maskgeration(imagepath, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = LightAttentionUNet() 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    dataset = GlacierTestDataset(imagepath_dict=imagepath)
   
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=False)
    
    masks = {}
    with torch.no_grad():
        for inputs_batch, tile_ids_batch in dataloader:
            inputs_batch = inputs_batch.to(device)
            outputs_batch = model(inputs_batch)
            pred_masks_batch = (outputs_batch.cpu().numpy() > 0.5).astype(np.uint8) * 255
            for i in range(len(tile_ids_batch)):
                tile_id = tile_ids_batch[i]
                binary_mask = pred_masks_batch[i].squeeze()
                masks[tile_id] = binary_mask
                
    return masks
