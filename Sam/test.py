import numpy as np
import os
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from matplotlib import cm
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


        # 读取NIfTI文件
image_file_path = "../Training/img/img0001.nii.gz"
label_file_path = "../Training/label/label0001.nii.gz"
img = nib.load(image_file_path)
label_img = nib.load(label_file_path)

# 获取图像和标签数据
image_data = img.get_fdata()
label_data = label_img.get_fdata()

num_slices = image_data.shape[-1]
output_folder = "1" 
os.makedirs(output_folder, exist_ok=True)
for i in range(100, num_slices):
    h, w = label_data[:, :, i].shape
    index = []
    image_box = []
    for l in range(1, 13) : 
        boxes = [h, w, 0, 0]
        for j in range(h):
            for k in range(w) : 
                if(label_data[j][k][i] == l) : 
                    boxes[0] = min(boxes[0], j)
                    boxes[1] = min(boxes[1], k)
                    boxes[2] = max(boxes[2], j)
                    boxes[3] = max(boxes[3], k)
        if(boxes[0] == h) : 
            continue
        l1 = boxes[2] - boxes[0]
        l2 = boxes[3] - boxes[1]
        image_box.append([boxes[1], boxes[0], boxes[3], boxes[2]])
        index.append(l)
    if(len(image_box)) : 
        image = image_data[:, :, i]
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print(image.shape)
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))

        ax[0].imshow(image,  cmap='gray')
        for box in image_box:
            show_box(np.array(box), ax[0])

        ax[1].imshow(label_data[:, :, i], cmap='jet')
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(output_folder, f'slice_{i + 1}.png')
        image_boxes = torch.tensor(image_box, device = sam.device)
        from segment_anything.utils.transforms import ResizeLongestSide
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        batched_input = [
        {
            'image': prepare_image(image, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image_boxes, image.shape[:2]),
            'original_size': image.shape[:2]
        },
        ]
        batched_output = sam(batched_input, multimask_output=False)
        for mask in batched_output[0]['masks']:
            show_mask(mask.cpu().numpy(), ax[0], random_color=True)
        plt.savefig(save_path)
        plt.close()
        # calc mDice 
        mDice = 0
        num = len(index)
        for T in range(num) : 
            label = np.zeros((h, w))
            for j in range(h):
                for k in range(w) : 
                    if(label_data[j][k][i] == index[T]) : 
                        label[j][k] = 1
            print(np.sum(label), np.sum(batched_output[0]['masks'][T].cpu().numpy().sum()))
            dice = np.sum(label * batched_output[0]['masks'][T].cpu().numpy()) * 2.0 / (np.sum(label) + np.sum(batched_output[0]['masks'][T].cpu().numpy()))
            print(dice)
            mDice += dice
        mDice /= len(index)
        print("mDice : ", mDice)
