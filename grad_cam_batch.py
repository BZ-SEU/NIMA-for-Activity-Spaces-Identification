# 作   者:BZ
# 开发时间:2024/12/18
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from utils_1 import GradCAM, show_cam_on_image
from NIMA_1 import NIMA
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chunk', default='childrens_activities', help='chunk')
args = parser.parse_args()


config = {
    "model_path": f"./trained_models/nima_{args.chunk}.pth",
    "excel_path": "./predict_results.xlsx",
    "image_dir": "./All_images",
    "activity": args.chunk,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

predict_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_txt = f"dataset_split/{config['activity']}_train_files.txt"
test_txt = f"dataset_split/{config['activity']}_test_files.txt"
log_file = 'logs/' + config['activity'] + '_log.txt'


def load_split_files():
    with open(train_txt, 'r') as f:
        train_files = {int(line.strip().split('.')[0]) for line in f}

    with open(test_txt, 'r') as f:
        test_files = {int(line.strip().split('.')[0]) for line in f}

    return train_files, test_files


# 加载模型
def load_model():
    model = NIMA()
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()
    return model.to(config["device"])


def get_new_img_size(height, width, img_min_side=384):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width


def predict_image(image_name):
    filename = int(image_name)
    df = pd.read_excel(config["excel_path"], sheet_name=config["activity"])

    row = df[df['filename'] == filename]
    if len(row) == 0:
        return None
    else:
        return {
            'split': row['split'].item(),
            'true_score': float(row['true_score'].item()),
            'pred_score': float(row['pred_score'].item()),
            'true_rank': int(row['true_rank'].item()),
            'pred_rank': int(row['pred_rank'].item())
        }


def cam_single_vgg(img, model, device):
    data_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)
    target_layers = [model.base_model.features]  # model.conv5
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=input_tensor[0].is_cuda)
    target_category = None
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    img = np.array(img)
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    return visualization


if __name__ == '__main__':
    torch.manual_seed(42)

    model = load_model()

    train_set, test_set = load_split_files()
    image_list = list(train_set | test_set)

    df = pd.read_excel(config["excel_path"], sheet_name=config["activity"])
    train_df = df[df['split'] == 'train']
    df['score_diff'] = abs(train_df['true_score'] - train_df['pred_score'])
    threshold_score_train = df['score_diff'].quantile(0.8)
    df['rank_diff'] = abs(train_df['true_rank'] - train_df['pred_rank'])
    threshold_rank_train = df['rank_diff'].quantile(0.8)
    test_df = df[df['split'] == 'test']
    df['score_diff'] = abs(test_df['true_score'] - test_df['pred_score'])
    threshold_score_test = df['score_diff'].quantile(0.8)
    df['rank_diff'] = abs(test_df['true_rank'] - test_df['pred_rank'])
    threshold_rank_test = df['rank_diff'].quantile(0.8)

    print('threshold_score_train=', threshold_score_train)
    print('threshold_rank_train =', threshold_rank_train)
    print('threshold_score_test =', threshold_score_test)
    print('threshold_rank_test  =', threshold_rank_test)
    with open(log_file, 'a') as file:
        file.write(f"Grad CAM finished.\n" +
                   f'threshold_score_train={threshold_score_train}\nthreshold_rank_train ={threshold_rank_train}\n' +
                   f'threshold_score_test ={threshold_score_test}\nthreshold_rank_test  ={threshold_rank_test}\n\n')

    # 进行遍历
    for img_name in tqdm(image_list):
        # print(img_name)
        image_path = config["image_dir"] + '/' + config["activity"] + '/' + str(int(img_name)) + '.jpg'
        image_origin = Image.open(image_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        image = predict_transform(image).unsqueeze(0).to(config["device"])
        img = Image.open(image_path).convert('RGB')
        img = img.resize((384, 384), Image.BICUBIC)

        # cam_fig = cam_single_vgg(img, model, config["device"])
        pdt_rst = predict_image(img_name)
        if pdt_rst is None:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(img)
        # im2 = ax2.imshow(cam_fig)
        plt.suptitle("Activity: " + config["activity"] + ", Filename: " + str(img_name) + ", Set: " + pdt_rst["split"],
                     y=0.95, fontsize=14, fontweight='bold', color='darkgreen')

        if (pdt_rst["split"] == 'train' and abs(pdt_rst['true_score'] - pdt_rst['pred_score']) > threshold_score_train)\
            or (pdt_rst["split"] == 'test' and abs(pdt_rst['true_score'] - pdt_rst['pred_score']) > threshold_score_test):
            score_color = 'red'
        else:
            score_color = 'black'
        if (pdt_rst["split"] == 'train' and abs(pdt_rst['true_rank'] - pdt_rst['pred_rank']) > threshold_rank_train)\
            or (pdt_rst["split"] == 'test' and abs(pdt_rst['true_rank'] - pdt_rst['pred_rank']) > threshold_rank_test):
            rank_color = 'red'
        else:
            rank_color = 'black'

        fig.text(0.2, 0.01,
                 f"true score = {pdt_rst['true_score']:.4f}\npred score = {pdt_rst['pred_score']:.4f}",
                 ha='left', fontsize=14, style='italic', color=score_color)
        fig.text(0.6, 0.01,
                 f"true rank = {pdt_rst['true_rank']}\npred rank = {pdt_rst['pred_rank']}",
                 ha='left', fontsize=14, style='italic', color=rank_color)
        plt.subplots_adjust(wspace=0.15,   bottom=0.15)
        save_dir = 'predict_result/' + config["activity"] + '/' + str(img_name) + '_cam.jpg'
        plt.savefig(save_dir)
        plt.close()

