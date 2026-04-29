import argparse
import cv2
import os
import torch
import numpy as np
from models.network_swinir import SwinIR as net

def main():
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root dataset folder (e.g., dataset/)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to SwinIR .pth model')
    
    # Configuration
    parser.add_argument('--scale', type=int, default=4, help='Scale factor: 2, 3, 4, 8')
    parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g., 400), None for whole image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap between tiles')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model (Classical SR with patch size 64)
    #model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
               # img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               # mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    # 1. Load Model (Real-World GAN SR - SwinIR Large)
    model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., 
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], # Increased to 9 blocks
                embed_dim=240,                      # Increased from 180 to 240
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8], # Increased to 8 heads
                mlp_ratio=2, 
                upsampler='nearest+conv',           # GAN uses nearest+conv
                resi_connection='3conv')            # GAN uses 3conv
    
    pretrained_model = torch.load(args.model_path)
    #param_key = 'params'
    #model.load_state_dict(pretrained_model[param_key] if param_key in pretrained_model.keys() else pretrained_model, strict=True)
    #model.eval().to(device)

    # Smarter loading: Check for GAN weights first, then classical weights
    if isinstance(pretrained_model, dict) and 'params_ema' in pretrained_model:
        weights = pretrained_model['params_ema']
    elif isinstance(pretrained_model, dict) and 'params' in pretrained_model:
        weights = pretrained_model['params']
    else:
        weights = pretrained_model
        
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)

    
    # 2. Find all images recursively in dataset_dir
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    img_paths = []
    for root, _, files in os.walk(args.dataset_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_paths.append(os.path.join(root, file))

    print(f"Found {len(img_paths)} images across classes. Starting in-place upscaling...")

    # 3. Process Images
    for idx, path in enumerate(img_paths):
        # Load and convert BGR to RGB
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read image {path}. Skipping...")
            continue
            
        img_lq = img.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1)) 
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # Pad to multiple of window size (8)
            _, _, h_old, w_old = img_lq.size()
            ws = 8
            h_pad = (h_old // ws + 1) * ws - h_old if h_old % ws != 0 else 0
            w_pad = (w_old // ws + 1) * ws - w_old if w_old % ws != 0 else 0
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            
            # Inference
            if args.tile is None:
                output = model(img_lq)
            else:
                output = tiled_inference(img_lq, model, args)
            
            # Crop padding and rescale
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # Post-process
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # RGB back to BGR
        output = (output * 255.0).round().astype(np.uint8)
        
        parent_dir = os.path.dirname(path)        # e.g. /content/dataset/classA/images
        grandparent_dir = os.path.dirname(parent_dir) # e.g. /content/dataset/classA
        file_name = os.path.basename(path)        # e.g. 001.png
        
        # Define the new folder name
        new_folder_name = "images_4x"
        save_dir = os.path.join(grandparent_dir, new_folder_name)
        
        # 2. CRITICAL: Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, file_name)

        # 3. Save and verify
        success = cv2.imwrite(save_path, output)
        
        if success:
            print(f"[{idx + 1}/{len(img_paths)}] Saved: {save_path}")
        else:
            print(f"!! FAILED to save: {save_path}. Check permissions or paths.")

def tiled_inference(img, model, args):
    b, c, h, w = img.size()
    tile = min(args.tile, h, w)
    stride = tile - args.tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    
    E = torch.zeros(b, c, h*args.scale, w*args.scale).type_as(img)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            E[..., h_idx*args.scale:(h_idx+tile)*args.scale, w_idx*args.scale:(w_idx+tile)*args.scale].add_(out_patch)
            W[..., h_idx*args.scale:(h_idx+tile)*args.scale, w_idx*args.scale:(w_idx+tile)*args.scale].add_(torch.ones_like(out_patch))
    return E.div_(W)

if __name__ == '__main__':
    main()
