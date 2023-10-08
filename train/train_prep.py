# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import argparse
import numpy as np
import imageio
import torch
import cv2
from tqdm import tqdm
from cloudvolume.lib import mkdir
from segment_anything_training import sam_model_registry


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--batch", "-b", type=int, required=True, default=1)

    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    return parser.parse_args()


args = get_args_parser()

sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
_ = sam.to(device=args.device)

mkdir(os.path.join(args.output, 'embed'))
mkdir(os.path.join(args.output, 'embed_interm_0'))
mkdir(os.path.join(args.output, 'embed_interm_1'))
mkdir(os.path.join(args.output, 'embed_interm_2'))
mkdir(os.path.join(args.output, 'embed_interm_3'))

images = imageio.volread(args.input)
batched_images = []
for b_i in tqdm(range(0, images.shape[0], args.batch)):
    b_j = min(b_i + args.batch, images.shape[0])
    single_input = []
    for i in range(b_i, b_j):
        image = images[i]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        single_input.append(image)
    batched_images.append(np.stack(single_input, axis=0))

for b in tqdm(range(len(batched_images))):
    imgs = batched_images[b]
    # print(imgs.shape)

    batched_input = []
    for i in range(len(imgs)):
        input_image = torch.as_tensor(imgs[i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
        batched_input.append(input_image)

    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = sam.image_encoder(input_images)
        # print(image_embeddings.shape, len(interm_embeddings))
    
    for i in range(len(image_embeddings)):
        image_embedding = image_embeddings[i].cpu().numpy()
        imageio.volwrite(os.path.join(args.output, 'embed', 
                                      '{:05d}.tif'.format(b * args.batch + i)), image_embedding)

        for l in range(len(interm_embeddings)):
            interm_embedding = interm_embeddings[l][i].cpu().numpy().transpose()
            # print(interm_embedding.shape)
            imageio.volwrite(os.path.join(args.output, 'embed_interm_{:d}'.format(l), 
                                          '{:05d}.tif'.format(b * args.batch + i)), interm_embedding)


'''
python train_prep.py -b 5 -i ./prepared_images/SNEMI.tif -o ./prepared_embedding/SNEMI \
    --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth --model-type vit_h
'''