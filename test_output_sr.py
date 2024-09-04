import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import torch.nn as nn
from models.model_agileir import agileir
from utils import utils_image as util

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_multadds(model, input_size=(3, 224, 224)):
    def hook_fn(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            input = input[0]
            batch_size = input.size(0)
            output_size = output.size()
            
            if isinstance(module, nn.Conv2d):
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
                output_elements = batch_size * output_size[1] * output_size[2] * output_size[3]
            else:  # nn.Linear
                kernel_ops = module.in_features
                output_elements = batch_size * output_size[1]
            
            multadds = kernel_ops * output_elements
            module.__multadds__ = multadds

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Perform a forward pass
    device = next(model.parameters()).device
    input = torch.rand(1, *input_size).to(device)
    model(input)

    # Sum up the MultAdds
    total_multadds = 0
    for module in model.modules():
        if hasattr(module, '__multadds__'):
            total_multadds += module.__multadds__

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return total_multadds

def format_number(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lightweight_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') 
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='./superresolution/agileir_v2/models/313600_G.pth')
    parser.add_argument('--folder_lq', type=str, default="testsets/Set5/LRbicx4", help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default="testsets/Set5/GTmod12", help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--input_image', type=str, default="input.png", help='input low-quality image')
    parser.add_argument('--output_image', type=str, default="output.png", help='output high-quality image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = agileir(upscale=2, img_size=64,
                   window_size=16, img_range=1., depths=[6] * 6,
                   embed_dim=180, num_heads=[6] * 6, key_dims=[16] * 6, mlp_ratio=2, upsampler='pixelshuffledirect')
    params = count_parameters(model)
    multadds = count_multadds(model)
    print(f"Parameters: {format_number(params)} ({params:,})")
    print(f"MultAdds: {format_number(multadds)} ({multadds:,})")

    model.eval()
    model = model.to(device)
    param_key_g = 'params'
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    # read input image
    img_lq = cv2.imread(args.input_image, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // 16 + 1) * 16 - h_old
        w_pad = (w_old // 16 + 1) * 16 - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old * args.scale, :w_old * args.scale]

    # save output image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(args.output_image, output)

    print(f"Input image saved as: {args.input_image}")
    print(f"Output image saved as: {args.output_image}")

if __name__ == '__main__':
    main()