#prepare
import PIL
import numpy as np
import os
import torch
import random
from tqdm import tqdm
os.chdir('/content/Painter/SegGPT/SegGPT_inference')

random.seed(42)

#function
def inference(input_image: list[str], prompt_img: list[str], prompt_trg: list[str], output_dir='/content'):
    assert len(prompt_img) == len(prompt_trg)
    prompt_size = len(prompt_img)
    path_to_input = os.path.abspath(input_image)
    !python seggpt_inference.py \
    --input_image $path_to_input \
    --prompt_image {' '.join(prompt_img)} \
    --prompt_target {' '.join(prompt_trg)} \
    --output_dir {output_dir}
    PIL.Image.open(os.path.abspath(output_dir)+'/output_'+os.path.basename(input_image)[:-3]+'png').point(lambda x: 255 if x else 0).convert('1').save(os.path.abspath(output_dir)+'/output_'+os.path.basename(input_image)[:-3]+'png')
    torch.cuda.empty_cache()

inference('/content/cryonuseg/test/images/Human_Larynx_02.tif', ['/content/cryonuseg/train/images/' + i for i in os.listdir('/content/cryonuseg/train/images')],
          ['/content/cryonuseg/train/labels/' + i for i in os.listdir('/content/cryonuseg/train/labels')])