from Sam_btcv import SamBTCV 
from segment_anything import sam_model_registry
from utils.prompt_gen import *
from utils.testing import test_model
from utils.dataset import BTCV2DSliceDataset, to_uint8_rgb, remove_pure_background, to_tensor
import torch
from torch.utils.data import DataLoader

model_root_path = "./models" # "./"

sam_model_pth = 'fin_sam_vit_h_4b8939.pth'
btcv_model_pth = "best_model.pth"

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    sam = sam_model_registry['default'](model_root_path + '/' + sam_model_pth)
    model = SamBTCV(sam, requires_classification=True)
    model.load_state_dict(torch.load(model_root_path + '/' + btcv_model_pth))
    model.to(device)

    preprocess = lambda images, labels: to_tensor(*to_uint8_rgb(*remove_pure_background(images, labels)))

    validation_set = BTCV2DSliceDataset(root_dir='./data', 
                                        json_file='./data/dataset_0.json', 
                                        type='validation', 
                                        preprocess=preprocess)
    
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: box_prompt(labels, device), 'box', model) 
    print(dice)
    print(mDice) 
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: point_prompt(labels, 1, device), 'point', model)
    print(dice)
    print(mDice)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: point_prompt(labels, 2, device), 'point', model)
    print(dice)
    print(mDice)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: point_prompt(labels, 3, device), 'point', model)
    print(dice)
    print(mDice)


if __name__ == '__main__':
    main()