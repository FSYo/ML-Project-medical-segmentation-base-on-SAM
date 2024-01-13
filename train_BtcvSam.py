import numpy as np
import os
import nibabel as nib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import sys
from matplotlib import cm
import json

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from Sam_btcv import SamBTCV

sam_checkpoint = "fin_sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_btcv = SamBTCV(sam, True)

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


# output_folder = "1" 
# os.makedirs(output_folder, exist_ok=True)

optimizer = torch.optim.Adam(sam_btcv.parameters(), lr=1e-6)
criterion = nn.CrossEntropyLoss()
num_epochs = 5
for epoch in range(5, 15):
    torch.save(sam_btcv.state_dict(), 'epoch_'+str(epoch)+sam_checkpoint)
    
    json_file_path = 'dataset_0.json'

    # 打开文件并加载 JSON 数据
    with open(json_file_path, 'r') as file:
        data = json.load(file)

        # validation test 
        # with torch.no_grad() :
        #     tot_mDice = 0
        #     tot_num = 0
        #     for tr_data in data["validation"] :
        #         image_file_path = "../Training/" + tr_data["img"]
        #         label_file_path = "../Training/" + tr_data["label"]
        #         img = nib.load(image_file_path)
        #         label_img = nib.load(label_file_path)

        #         # 获取图像和标签数据
        #         image_data = img.get_fdata()
        #         label_data = label_img.get_fdata()

        #         num_slices = image_data.shape[-1]
        #         for i in range(num_slices):
        #             h, w = label_data[:, :, i].shape
        #             index = []
        #             image_box = []
        #             for l in range(1, 13) : 
        #                 boxes = [h, w, 0, 0]
        #                 for j in range(h):
        #                     for k in range(w) : 
        #                         if(label_data[j][k][i] == l) : 
        #                             boxes[0] = min(boxes[0], j)
        #                             boxes[1] = min(boxes[1], k)
        #                             boxes[2] = max(boxes[2], j)
        #                             boxes[3] = max(boxes[3], k)
        #                 if(boxes[0] == h) : 
        #                     continue
        #                 l1 = boxes[2] - boxes[0]
        #                 l2 = boxes[3] - boxes[1]
        #                 image_box.append([boxes[1], boxes[0], boxes[3], boxes[2]])
        #                 index.append(l)
        #             if(len(image_box)) : 
        #                 image = image_data[:, :, i]
        #                 image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        #                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #                 image_boxes = torch.tensor(image_box, device = sam_btcv.device)
        #                 from segment_anything.utils.transforms import ResizeLongestSide
        #                 resize_transform = ResizeLongestSide(sam_btcv.image_encoder.img_size)
        #                 batched_input = [
        #                 {
        #                     'image': prepare_image(image, resize_transform, sam_btcv),
        #                     'boxes': resize_transform.apply_boxes_torch(image_boxes, image.shape[:2]),
        #                     'original_size': image.shape[:2]
        #                 },
        #                 ]
        #                 batched_output = sam_btcv(batched_input, multimask_output=False)
        #                 mask_threshold = 0.0
        #                 mDice = 0
        #                 num = len(index)
        #                 for T in range(num) : 
        #                     label = np.zeros((h, w))
        #                     for j in range(h):
        #                         for k in range(w) : 
        #                             if(label_data[j][k][i] == index[T]) : 
        #                                 label[j][k] = 1
        #                     answer = (batched_output[0]['masks'][T] > mask_threshold).cpu().numpy()
        #                     dice = np.sum(label * answer) * 2.0 / (np.sum(label) + np.sum(answer))
        #                     print(dice, end=' ')
        #                     mDice += dice
        #                 mDice /= len(index)
        #                 tot_mDice += mDice
        #                 tot_num += 1
                        
        #                 print("\nmDice : ", mDice)
        #     print("Epoch : ", epoch)
        #     print("average mDice in validation test : ", tot_mDice / tot_num)    

        SUM_LOSS = 0
        SUM_DICE = 0
        NUM = 0
        # datas = []
        
        # for tr_data in data["training"] :
        #     datas.append(("../Training/" + tr_data["img"], "../Training/" + tr_data["label"]))
        


        for tr_data in data["training"] :
            image_file_path = "../Training/" + tr_data["img"]
            label_file_path = "../Training/" + tr_data["label"]
            img = nib.load(image_file_path)
            label_img = nib.load(label_file_path)

            # 获取图像和标签数据
            image_data = img.get_fdata()
            label_data = label_img.get_fdata()

            num_slices = image_data.shape[-1]
            for i in range(num_slices):
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

                    image_boxes = torch.tensor(image_box, device = sam_btcv.device)
                    from segment_anything.utils.transforms import ResizeLongestSide
                    resize_transform = ResizeLongestSide(sam_btcv.image_encoder.img_size)
                    batched_input = [
                    {
                        'image': prepare_image(image, resize_transform, sam_btcv),
                        'boxes': resize_transform.apply_boxes_torch(image_boxes, image.shape[:2]),
                        'original_size': image.shape[:2]
                    },
                    ]
                    batched_output = sam_btcv(batched_input, multimask_output=False)
                    mask_threshold = 0.0
                    
                    mDice = 0
                    num = len(index)

                    optimizer.zero_grad()
                    sum_loss = torch.tensor(0.0)
                    sum_loss = sum_loss.to('cuda')
                    # print('image ', i, end = ':')
                    for T in range(num) : 
                        label = np.zeros((h, w))
                        for j in range(h):
                            for k in range(w) : 
                                if(label_data[j][k][i] == index[T]) : 
                                    label[j][k] = 1

                        res = torch.sigmoid(batched_output[0]['masks'][T][0])
                        # print(res.shape)
                        # print(label.shape)
                # print(label)
                     # print(res)
                        label = torch.tensor(label)
                        label = label.to('cuda')
                        

                        smooth_dice = torch.sum(label * res) * 2 / (torch.sum(label) + torch.sum(res))
                        sum_loss += 1 - smooth_dice
 
                        answer = (batched_output[0]['masks'][T] > mask_threshold).cpu().numpy()

                        label = label.to('cpu')
                        label = label.numpy()
                        dice = np.sum(label * answer) * 2.0 / (np.sum(label) + np.sum(answer))
                        print(dice, end=' ')
                        mDice += dice
                    print("skip is here", sum_loss)
                    sum_loss.backward()
                    optimizer.step()
                    mDice /= len(index)
                    print("\nmDice : ", mDice)
                    print("LOSS : ", sum_loss)
                    
                    SUM_LOSS += sum_loss
                    SUM_DICE += mDice
                    NUM += 1 
        print("ave loss and ave dice : ", SUM_LOSS / NUM, SUM_DICE / NUM)
                    
torch.save(sam_btcv.state_dict(), 'epoch'+str(num_epochs)+sam_checkpoint)