# pip install torch torchvision numpy pillow tifffile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import re 


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

    def __init__(self, in_channels=5, out_channels=4):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(256, 256, 128); self.dec4 = conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(128, 128, 64); self.dec3 = conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(64, 64, 32); self.dec2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(32, 32, 16); self.dec1 = conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))
        d4 = self.upconv4(bottleneck); s4 = self.ag4(g=d4, s=e4); d4 = torch.cat((d4, s4), dim=1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); s3 = self.ag3(g=d3, s=e3); d3 = torch.cat((d3, s3), dim=1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); s2 = self.ag2(g=d2, s=e2); d2 = torch.cat((d2, s2), dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); s1 = self.ag1(g=d1, s=e1); d1 = torch.cat((d1, s1), dim=1); d1 = self.dec1(d1)
        
        return self.final_conv(d1)

class GlacierTestDataset(Dataset):

    def __init__(self, imagepath_dict, bands=['Band1', 'Band2', 'Band3', 'Band4', 'Band5']):
        self.imagepath_dict = imagepath_dict
        self.bands = sorted(bands)
        
        ref_band_folder = None
        for band in self.bands:
            if band in self.imagepath_dict and os.path.exists(self.imagepath_dict[band]):
                ref_band_folder = self.imagepath_dict[band]
                break
        
        if ref_band_folder is None:
            raise FileNotFoundError("No valid band folders found in imagepath_dict.")
            
        self.filenames = sorted([f for f in os.listdir(ref_band_folder) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        input_bands = []
        for band in self.bands:
            band_folder = self.imagepath_dict[band]
            file_path = os.path.join(band_folder, filename)
            try:
                band_arr = np.array(Image.open(file_path), dtype=np.float32)
                if band_arr.ndim == 3: band_arr = band_arr[..., 0]
                input_bands.append(band_arr)
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}, returning empty.")
                h, w = Image.open(os.path.join(self.imagepath_dict[self.bands[0]], self.filenames[0])).size
                input_bands.append(np.zeros((w, h), dtype=np.float32)) 
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                h, w = Image.open(os.path.join(self.imagepath_dict[self.bands[0]], self.filenames[0])).size
                input_bands.append(np.zeros((w, h), dtype=np.float32))

        input_tensor = np.stack(input_bands, axis=0)
        
        tile_id = os.path.splitext(filename)[0].replace('img', '')
        
        return torch.from_numpy(input_tensor), tile_id


def maskgeration(imagepath, model_path):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LightAttentionUNet(in_channels=5, out_channels=4) 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure 'model_path' points to a valid 4-class LightAttentionUNet model.")
        return {}
        
    model.to(device)
    model.eval()
    
    dataset = GlacierTestDataset(imagepath_dict=imagepath)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=False)
    
    masks = {}
    print(f"Starting prediction for {len(dataset)} images...")
    with torch.no_grad():
        for inputs_batch, tile_ids_batch in dataloader:
            inputs_batch = inputs_batch.to(device)
            outputs_batch = model(inputs_batch)
            pred_indices_batch = torch.argmax(outputs_batch, dim=1).cpu().numpy().astype(np.uint8)
            for i in range(len(tile_ids_batch)):
                tile_id = tile_ids_batch[i]
                indices_mask = pred_indices_batch[i] 
                final_mask = np.zeros_like(indices_mask, dtype=np.uint8)
                final_mask[indices_mask == 1] = 85  
                final_mask[indices_mask == 2] = 170 
                final_mask[indices_mask == 3] = 255 
                
                masks[tile_id] = final_mask
                
    print(f"Prediction complete. Generated {len(masks)} masks.")
    return masks