import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import *

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
import os
import pdb
from tqdm import tqdm

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt = config.pretrained_model
model_config = config.config_file

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# load dinov2 model
processor_dinov2 = AutoImageProcessor.from_pretrained('path/dinov2')
model_dinov2 = AutoModel.from_pretrained('path/dinov2').to(device)

def calculate_area(file_name):
    return int(file_name.split('_')[3].split('.')[0])

def get_mask_bbox(mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h
    return None

def get_crop_image(bg_image, mask):
    img_height, img_width, _ = bg_image.shape

    bbox = get_mask_bbox(mask)

    x, y, w, h = bbox
    x_new = int(x - 0.1 * w)
    y_new = int(y - 0.1 * h)
    w_new = int(w * 1.2)
    h_new = int(h * 1.2)

    x_new = max(0, x_new)
    y_new = max(0, y_new)
    w_new = min(w_new, img_width - x_new)
    h_new = min(h_new, img_height - y_new)

    crop_img = bg_image[y_new:y_new + h_new, x_new:x_new + w_new]
    return crop_img

def convert_cv2_to_pil(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def get_image_features(image):
    if isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image)

    elif isinstance(image, np.ndarray):
        image = convert_cv2_to_pil(image)

    if not isinstance(image, Image.Image):
        raise ValueError("invalid input")

    with torch.no_grad():
        inputs = processor_dinov2(images=image, return_tensors="pt").to(device)
        outputs = model_dinov2(**inputs)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)

    return image_features

def encode_images_in_directory(directory_path):
    image_features_dict = {}
    for image_file in os.listdir(directory_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, image_file)
            image_features = get_image_features(image_path)
            image_features_dict[image_file] = image_features
    return image_features_dict

def find_most_similar_image(crop_image, image_features_dict, reference_files):
    given_image_features = get_image_features(crop_image)
    cos = nn.CosineSimilarity(dim=0)
    max_score = 0
    most_similar_image = None

    for image_file in reference_files:
        stored_features = image_features_dict[image_file]
        score = cos(given_image_features[0], stored_features[0]).item()
        score = (score + 1) / 2
        if score > max_score:
            max_score = score
            most_similar_image = image_file

    return most_similar_image, max_score

def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])
    transformed = transform(image=image.astype(np.uint8), mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image)

    # collage aug
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask)
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) )
    return item

def process_double_pairs(ref_image1, ref_mask1, ref_image2, ref_mask2, tar_image, tar_mask):
    ref_box_yyxx1 = get_bbox_from_mask(ref_mask1)
    ref_box_yyxx2 = get_bbox_from_mask(ref_mask2)

    ref_mask_31 = np.stack([ref_mask1, ref_mask1, ref_mask1], -1)
    masked_ref_image1 = ref_image1 * ref_mask_31 + np.ones_like(ref_image1) * 255 * (1 - ref_mask_31)

    ref_mask_32 = np.stack([ref_mask2, ref_mask2, ref_mask2], -1)
    masked_ref_image2 = ref_image2 * ref_mask_32 + np.ones_like(ref_image2) * 255 * (1 - ref_mask_32)

    y11, y12, x11, x12 = ref_box_yyxx1
    masked_ref_image1 = masked_ref_image1[y11:y12, x11:x12, :]
    ref_mask1 = ref_mask1[y11:y12, x11:x12]

    y21, y22, x21, x22 = ref_box_yyxx2
    masked_ref_image2 = masked_ref_image2[y21:y22, x21:x22, :]
    ref_mask2 = ref_mask2[y21:y22, x21:x22]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image1, ref_mask1 = expand_image_mask(masked_ref_image1, ref_mask1, ratio=ratio)
    masked_ref_image2, ref_mask2 = expand_image_mask(masked_ref_image2, ref_mask2, ratio=ratio)

    ref_mask_31 = np.stack([ref_mask1, ref_mask1, ref_mask1], -1)
    ref_mask_32 = np.stack([ref_mask2, ref_mask2, ref_mask2], -1)

    masked_ref_image1 = pad_to_square(masked_ref_image1, pad_value=255, random=False)
    masked_ref_image1 = cv2.resize(masked_ref_image1, (224, 224)).astype(np.uint8)

    masked_ref_image2 = pad_to_square(masked_ref_image2, pad_value=255, random=False)
    masked_ref_image2 = cv2.resize(masked_ref_image2, (224, 224)).astype(np.uint8)

    ref_mask_31 = pad_to_square(ref_mask_31 * 255, pad_value=0, random=False)
    ref_mask_31 = cv2.resize(ref_mask_31, (224, 224)).astype(np.uint8)
    ref_mask1 = ref_mask_31[:, :, 0]

    ref_mask_32 = pad_to_square(ref_mask_32 * 255, pad_value=0, random=False)
    ref_mask_32 = cv2.resize(ref_mask_32, (224, 224)).astype(np.uint8)
    ref_mask2 = ref_mask_32[:, :, 0]

    ref_image_collage1 = sobel(masked_ref_image1, ref_mask1 / 255)
    ref_image_collage2 = sobel(masked_ref_image2, ref_mask2 / 255)

    combined_hf1 = cv2.addWeighted(ref_image_collage1, 0.1, ref_image_collage2, 0.9, 0)
    combined_hf2 = cv2.addWeighted(ref_image_collage1, 0.9, ref_image_collage2, 0.1, 0)

    vis_HFmap = cv2.hconcat([ref_image_collage2, ref_image_collage1, combined_hf2, combined_hf1])

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])

    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    ref_image_collage1 = cv2.resize(combined_hf1, (x2 - x1, y2 - y1))
    ref_image_collage2 = cv2.resize(combined_hf2, (x2 - x1, y2 - y1))
    ref_mask1 = cv2.resize(ref_mask1.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask1 = (ref_mask1 > 128).astype(np.uint8)
    ref_mask2 = cv2.resize(ref_mask2.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask2 = (ref_mask2 > 128).astype(np.uint8)

    collage1 = cropped_target_image.copy()
    collage1[y1:y2, x1:x2, :] = ref_image_collage1
    collage_mask1 = cropped_target_image.copy() * 0.0
    collage_mask1[y1:y2, x1:x2, :] = 1.0

    collage2 = cropped_target_image.copy()
    collage2[y1:y2, x1:x2, :] = ref_image_collage2
    collage_mask2 = cropped_target_image.copy() * 0.0
    collage_mask2[y1:y2, x1:x2, :] = 1.0

    H1, W1 = collage1.shape[0], collage1.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage1 = pad_to_square(collage1, pad_value=0, random=False).astype(np.uint8)
    collage_mask1 = pad_to_square(collage_mask1, pad_value=-1, random=False).astype(np.uint8)
    collage2 = pad_to_square(collage2, pad_value=0, random=False).astype(np.uint8)
    collage_mask2 = pad_to_square(collage_mask2, pad_value=-1, random=False).astype(np.uint8)

    H2, W2 = collage1.shape[0], collage1.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(np.float32)
    collage1 = cv2.resize(collage1, (512, 512)).astype(np.float32)
    collage_mask1 = (cv2.resize(collage_mask1, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)
    collage2 = cv2.resize(collage2, (512, 512)).astype(np.float32)
    collage_mask2 = (cv2.resize(collage_mask2, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image1 = masked_ref_image1 / 255
    masked_ref_image2 = masked_ref_image2 / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage1 = collage1 / 127.5 - 1.0
    collage1 = np.concatenate([collage1, collage_mask1[:, :, :1]], -1)
    collage2 = collage2 / 127.5 - 1.0
    collage2 = np.concatenate([collage2, collage_mask2[:, :, :1]], -1)

    item1 = dict(ref=masked_ref_image1.copy(), jpg=cropped_target_image.copy(), hint=collage1.copy(),
                 extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array(tar_box_yyxx_crop))
    item2 = dict(ref=masked_ref_image2.copy(), jpg=cropped_target_image.copy(), hint=collage2.copy(),
                 extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array(tar_box_yyxx_crop))

    return item1, item2, vis_HFmap


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5  # maigin_pixel

    if W1 == H1:
        tar_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
    return gen_image

def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale=3):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:, :, :-1]
    hint_mask = item['hint'][:, :, -1] * 255
    hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
    ref = cv2.resize(ref.astype(np.uint8), (512, 512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg']
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda()
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(ref.copy()).float().cuda()
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H, W = 512, 512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(clip_input)]}
    un_cond = {"c_concat": None if guess_mode else [control],
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1, 3, 224, 224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1  # gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False  # gr.Checkbox(label='Guess Mode', value=False)
    # detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50  # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0  # gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples,
                                  'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()  # .clip(0, 255).astype(np.uint8)

    result = x_samples[0][:, :, ::-1]
    result = np.clip(result, 0, 255)

    pred = x_samples[0]
    pred = np.clip(pred, 0, 255)[1:, :, :]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)
    return gen_image

def inference_double_image(ref_image1, ref_mask1, ref_image2, ref_mask2, tar_image, tar_mask, guidance_scale=3):
    item1, item2, vis_HFmap = process_double_pairs(ref_image1, ref_mask1, ref_image2, ref_mask2, tar_image, tar_mask)

    results = []
    for item in [item1, item2]:
        ref = item['ref'] * 255
        tar = item['jpg'] * 127.5 + 127.5
        hint = item['hint'] * 127.5 + 127.5

        hint_image = hint[:, :, :-1]
        hint_mask = item['hint'][:, :, -1] * 255
        hint_mask = np.stack([hint_mask, hint_mask, hint_mask], -1)
        ref = cv2.resize(ref.astype(np.uint8), (512, 512))

        seed = random.randint(0, 65535)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        ref = item['ref']
        tar = item['jpg']
        hint = item['hint']
        num_samples = 1

        control = torch.from_numpy(hint.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        clip_input = torch.from_numpy(ref.copy()).float().cuda()
        clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

        guess_mode = False
        H, W = 512, 512

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(clip_input)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([torch.zeros((1, 3, 224, 224))] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        # ====
        num_samples = 1  # gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
        image_resolution = 512  # gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
        strength = 1  # gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
        guess_mode = False  # gr.Checkbox(label='Guess Mode', value=False)
        # detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
        ddim_steps = 50  # gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
        scale = guidance_scale  # gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
        seed = -1  # gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
        eta = 0.0  # gr.Number(label="eta (DDIM)", value=0.0)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples,
                                      'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()  # .clip(0, 255).astype(np.uint8)

        result = x_samples[0][:, :, ::-1]
        result = np.clip(result, 0, 255)

        pred = x_samples[0]
        pred = np.clip(pred, 0, 255)[1:, :, :]
        sizes = item['extra_sizes']
        tar_box_yyxx_crop = item['tar_box_yyxx_crop']
        gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop)

        results.append(gen_image)

    return results, vis_HFmap

if __name__ == '__main__':
    # foreground image path (FG), also known as Abnormal Cell Bank
    reference_image_dir = 'examples/reference'
    # background image path (BG)
    image_dir = 'examples/background'
    # background mask path (BG_mask)
    mask_dir = 'examples/mask'
    # type record path (prepare on your own)
    TypeAware_txt_file_path = 'examples/inference_results_TypeAware.txt'
    # save path
    composition_dir_initial = 'examples/gen'

    os.makedirs(composition_dir_initial, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    fussion_mode = True

    type_aware_dict = {}
    with open(TypeAware_txt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            reference_prefix = parts[0]
            cell_type = int(parts[1])

            if cell_type not in [0, 1]:
                cell_type = random.choice([0, 1])

            type_aware_dict[reference_prefix] = cell_type

    reference_files_dict = {}
    for f in os.listdir(reference_image_dir):
        file_category = f.split('_')[0]

        reference_prefix = os.path.splitext(f)[0]

        cell_type = type_aware_dict.get(reference_prefix, None)

        if cell_type is not None:
            reference_files_dict.setdefault((file_category, cell_type), []).append(f)

    image_features_dict = encode_images_in_directory(reference_image_dir)

    for image_file in tqdm(image_files, desc='Processing background images'):

        if image_file not in os.listdir(composition_dir_initial):

            file_prefix = os.path.splitext(image_file)[0]

            mask_folder = os.path.join(mask_dir, file_prefix)
            if not os.listdir(mask_folder):
                print(mask_folder)
                continue

            mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.jpg')]

            bg_image_path_initial = os.path.join(image_dir, image_file)

            for i, mask_file in enumerate(mask_files):
                try:
                    if random.choices([True, False], weights=[1, 0])[0]:
                        mask_file_category = mask_file.split('_')[2]
                        mask_file_area = int(mask_file.split('_')[3])
                        new_mask_name = "_".join([mask_file_category, file_prefix, str(mask_file_area)])
                        mask_file_cell_type = type_aware_dict.get(new_mask_name, random.choice([0, 1]))

                        reference_files = reference_files_dict.get((mask_file_category, mask_file_cell_type), [])

                        if fussion_mode:

                            bg_mask_path = os.path.join(mask_folder, mask_file)
                            crop_image = get_crop_image(cv2.imread(bg_image_path_initial), cv2.imread(bg_mask_path))

                            most_similar_image, score = find_most_similar_image(crop_image, image_features_dict, reference_files)
                            print('DINOv2 Score:',score)

                            closest_reference_file = random.choices(sorted(reference_files, key=lambda x: abs(calculate_area(x) - mask_file_area))[:3], weights=[0.65, 0.25, 0.1])[0]
                            # closest_reference_file = random.choice(reference_files)

                            second_closest_reference_file = most_similar_image

                            min_reference_image_file = closest_reference_file
                            min_reference_image_path = os.path.join(reference_image_dir, min_reference_image_file)

                            second_min_reference_image_file = second_closest_reference_file
                            second_min_reference_image_path = os.path.join(reference_image_dir, second_min_reference_image_file)

                            composition_path_initial = os.path.join(composition_dir_initial, image_file)

                            image1 = cv2.imread(min_reference_image_path, cv2.IMREAD_UNCHANGED)
                            mask1 = (image1[:, :, -1] > 128).astype(np.uint8)
                            image1 = image1[:, :, :-1]
                            image1 = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB)
                            ref_image1 = image1
                            ref_mask1 = mask1

                            image2 = cv2.imread(second_min_reference_image_path, cv2.IMREAD_UNCHANGED)
                            mask2 = (image2[:, :, -1] > 128).astype(np.uint8)
                            image2 = image2[:, :, :-1]
                            image2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2RGB)
                            ref_image2 = image2
                            ref_mask2 = mask2

                            # background image
                            back_image_initial = cv2.imread(bg_image_path_initial).astype(np.uint8)
                            back_image_initial = cv2.cvtColor(back_image_initial, cv2.COLOR_BGR2RGB)

                            # background mask
                            tar_mask = cv2.imread(bg_mask_path)[:, :, 0] > 128
                            tar_mask = tar_mask.astype(np.uint8)

                            scene_mask = cv2.cvtColor(tar_mask * 255, cv2.COLOR_GRAY2BGR)


                            gen_images, vis_HFmap = inference_double_image(ref_image1, ref_mask1, ref_image2, ref_mask2, back_image_initial.copy(), tar_mask)

                            cv2.imwrite(composition_path_initial, gen_images[0][:, :, ::-1])

                            bg_image_path_initial = composition_path_initial


                except Exception as e:
                    print(f"Error processing mask file {mask_file} in image {image_file}: {e}")








