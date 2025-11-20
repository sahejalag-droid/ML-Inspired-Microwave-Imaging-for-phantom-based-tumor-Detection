# generate_phantoms.py
# Tuned synthetic phantom generator for clearer tumor vs no_tumor distinction
# - stronger tumor contrast
# - larger tumor sizes
# - lower noise
# - balanced classes by default
#
# Usage: python generate_phantoms.py
# Requires: numpy, opencv-python

import os, csv
import numpy as np
import cv2
from math import pi, cos, sin

# ---------------- CONFIG (tuned) ----------------
OUT_DIR = os.path.join("data", "synthetic")
N_IMAGES = 1200                # total images to generate
IMG_SIZE = (256, 256)          # (h, w)
FORCE_BALANCE = True           # produce ~equal tumor/no_tumor
TUMOR_PROB = 0.75              # used if FORCE_BALANCE=False
SEED = 123
SAVE_VERBOSE = True

# Tumor contrast (0..1). Increased so tumors are clearly brighter than background
TUMOR_CONTRAST_MIN = 0.90
TUMOR_CONTRAST_MAX = 1.00

# Tissue texture intensity (smaller => cleaner image)
TISSUE_TEXTURE_INT = 0.02

# Speckle noise scale (multiplicative). Lower = cleaner images
SPECKLE_SCALE_MIN = 0.005
SPECKLE_SCALE_MAX = 0.015

# Max blur kernel (odd int). Keep small so edges are preserved.
MAX_BLUR = 3

# irregular tumor chance and params: make irregular more pronounced
IRREGULAR_PROB = 0.30
IRREGULAR_MIN = 0.25
IRREGULAR_MAX = 0.55

# Tumor size relative radii (increase so tumors are larger)
MIN_RADIUS_RATIO = 0.06
MAX_RADIUS_RATIO = 0.15
# ----------------------------------------

np.random.seed(SEED)

def ensure_dirs(out_dir):
    images = os.path.join(out_dir, "images")
    masks  = os.path.join(out_dir, "masks")
    os.makedirs(images, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    return images, masks

def create_arm_background(size, bone=True):
    h, w = size
    bg = np.zeros((h,w), dtype=np.float32)
    cx = w//2 + np.random.randint(-8,9)
    cy = h//2 + np.random.randint(-6,7)
    axes = (int(w*0.36 + np.random.randint(-6,6)), int(h*0.26 + np.random.randint(-4,4)))
    angle = np.random.uniform(-8,8)
    cv2.ellipse(bg, (cx,cy), axes, angle, 0, 360, color=0.6, thickness=-1)
    # radial gradient for subtle shading
    yy,xx = np.indices((h,w))
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    grad = 1.0 - (dist / (max(axes)+1))
    grad = np.clip(grad, 0, 1)
    bg = np.clip(bg * (0.88 + 0.22*grad), 0, 1)
    if bone and np.random.rand() < 0.5:
        bone_axes = (int(axes[0]*0.24), int(axes[1]*0.38))
        bcx = cx + np.random.randint(-8,9)
        bcy = cy + np.random.randint(-4,5)
        cv2.ellipse(bg, (bcx,bcy), bone_axes, angle + np.random.uniform(-3,3), 0, 360, color=0.92, thickness=-1)
    return bg

def draw_circle(mask, center, r):
    cv2.circle(mask, center, int(r), color=1, thickness=-1)
    return mask

def draw_ellipse(mask, center, axes, angle=0):
    cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
    return mask

def draw_irregular(mask, center, base_r, irregularity=0.35, spikes=0.15, n_points=140):
    h,w = mask.shape
    cx,cy = center
    angles = np.linspace(0, 2*pi, n_points, endpoint=False)
    radii = base_r * (1 + irregularity * np.random.randn(n_points))
    idx = np.random.choice(n_points, size=int(n_points*spikes), replace=False)
    radii[idx] *= (1 + 0.8 * np.random.rand(len(idx)))
    # smoother convolution window to avoid jaggedness
    k = 9
    radii = np.convolve(radii, np.ones(k)/k, mode='same')
    pts = []
    for a,r in zip(angles, radii):
        x = int(np.clip(cx + r * cos(a), 0, w-1))
        y = int(np.clip(cy + r * sin(a), 0, h-1))
        pts.append([x,y])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5,5), 0)
    _, mask_bin = cv2.threshold(mask, 0.2, 1.0, cv2.THRESH_BINARY)
    return mask_bin

def add_tissue_texture(img, intensity=TISSUE_TEXTURE_INT):
    h,w = img.shape
    noise = np.random.randn(h,w).astype(np.float32)
    k = max(11, min(h//4, w//4))
    if k % 2 == 0: k += 1
    tex = cv2.GaussianBlur(noise, (k,k), 0)
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    img = img + intensity * (tex - 0.5)
    return img

def add_speckle(img, scale):
    noise = 1 + scale * np.random.randn(*img.shape)
    return img * noise

def imaging_transforms(img, max_blur=MAX_BLUR):
    k = np.random.randint(1, max_blur*2+2)
    if k % 2 == 0: k += 1
    img = cv2.GaussianBlur(img, (k,k), 0)
    alpha = np.random.uniform(0.96, 1.04)
    beta  = np.random.uniform(-0.03, 0.03)
    img = np.clip(img * alpha + beta, 0, 1)
    return img

def random_rotate(img, mask, max_angle=6):
    angle = np.random.uniform(-max_angle, max_angle)
    h,w = img.shape
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    img_r = cv2.warpAffine((img*255).astype(np.uint8), M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_r = cv2.warpAffine((mask*255).astype(np.uint8), M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r.astype(np.float32)/255.0, (mask_r>127).astype(np.uint8)

def generate_one(size, tumor_present, forced_ttype=None):
    h,w = size
    bg = create_arm_background(size)
    mask = np.zeros((h,w), dtype=np.uint8)
    label = "no_tumor"
    if tumor_present:
        cx = w//2 + np.random.randint(-int(w*0.06), int(w*0.06)+1)
        cy = h//2 + np.random.randint(-int(h*0.06), int(h*0.06)+1)
        # pick type with irregular more visible
        if forced_ttype:
            ttype = forced_ttype
        else:
            ttype = np.random.choice(['circle','ellipse','elongated','irregular'],
                                     p=[0.26,0.26,0.12,0.36])
        base_r = np.random.randint(int(min(h,w)*MIN_RADIUS_RATIO), int(min(h,w)*MAX_RADIUS_RATIO))
        if ttype == 'circle':
            mask = draw_circle(mask, (cx,cy), base_r)
            label = 'tumor_circle'
        elif ttype == 'ellipse':
            axes = (int(base_r*1.2), int(base_r*0.8))
            mask = draw_ellipse(mask, (cx,cy), axes, angle=np.random.uniform(0,180))
            label = 'tumor_ellipse'
        elif ttype == 'elongated':
            axes = (int(base_r*1.9), int(base_r*0.55))
            mask = draw_ellipse(mask, (cx,cy), axes, angle=np.random.uniform(0,180))
            label = 'tumor_elongated'
        else:
            mask = draw_irregular(mask, (cx,cy), base_r, irregularity=np.random.uniform(IRREGULAR_MIN, IRREGULAR_MAX),
                                  spikes=np.random.uniform(0.08,0.30))
            label = 'tumor_irregular'
        # tumor contrast stronger: sample from configured range
        tumor_intensity = np.random.uniform(TUMOR_CONTRAST_MIN, TUMOR_CONTRAST_MAX)
        bg = np.where(mask==1, tumor_intensity, bg)
    # texture & noise (reduced)
    img = add_tissue_texture(bg, intensity=TISSUE_TEXTURE_INT)
    speckle_scale = np.random.uniform(SPECKLE_SCALE_MIN, SPECKLE_SCALE_MAX)
    img = add_speckle(img, speckle_scale)
    img = imaging_transforms(img, max_blur=MAX_BLUR)
    img, mask = random_rotate(img, mask, max_angle=6)
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    mask_u8 = (mask * 255).astype(np.uint8)
    return img_u8, mask_u8, label

def generate_dataset(out_dir, n_images, force_balance=FORCE_BALANCE, tumor_prob=TUMOR_PROB):
    images_dir, masks_dir = ensure_dirs(out_dir)
    csv_path = os.path.join(out_dir, "labels.csv")

    # prepare balanced/unbalanced plan
    if force_balance:
        half = n_images // 2
        labels_plan = [False]*half + [True]*(n_images-half)
        np.random.shuffle(labels_plan)
    else:
        labels_plan = [np.random.rand() < tumor_prob for _ in range(n_images)]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image','mask','label'])
        for i, tumor_present in enumerate(labels_plan):
            img, mask, lbl = generate_one(IMG_SIZE, tumor_present)
            fname = f"img_{i:04d}.png"
            mname = f"mask_{i:04d}.png"
            cv2.imwrite(os.path.join(images_dir, fname), img)
            cv2.imwrite(os.path.join(masks_dir, mname), mask)
            writer.writerow([os.path.join("images", fname), os.path.join("masks", mname), lbl])
            if SAVE_VERBOSE and (i % 50 == 0):
                print(f"[{i}/{n_images}] saved {fname} label={lbl}")
    print("Done. Dataset:", out_dir)

if __name__ == "__main__":
    print("Generating dataset at:", OUT_DIR)
    generate_dataset(OUT_DIR, N_IMAGES, force_balance=FORCE_BALANCE, tumor_prob=TUMOR_PROB)
