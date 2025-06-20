# 作   者:BZ
# 开发时间:2025/3/3
import os
import shutil
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from NIMA_1 import NIMA
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chunk', default='childrens_activities', help='chunk')
args = parser.parse_args()

# 配置参数
predict_config = {
    "model_path": f"./trained_models/nima_{args.chunk}.pth",
    "excel_path": "./All_labels/Result0316.xlsx",  # Excel文件路径
    "image_dir": "./All_images",  # 图片文件夹路径
    "activity": args.chunk,
    "output_excel": f"./predict_results.xlsx",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

predict_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_txt = f"dataset_split/{predict_config['activity']}_train_files.txt"
test_txt = f"dataset_split/{predict_config['activity']}_test_files.txt"
log_file = 'logs/' + predict_config['activity'] + '_log.txt'

def load_model():
    model = NIMA()
    model.load_state_dict(torch.load(predict_config["model_path"]))
    model.eval()
    return model.to(predict_config["device"])


def load_split_files():
    with open(train_txt, 'r') as f:
        train_files = {int(line.strip().split('.')[0]) for line in f}

    with open(test_txt, 'r') as f:
        test_files = {int(line.strip().split('.')[0]) for line in f}

    return train_files, test_files

def predict_image(model, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = predict_transform(image).unsqueeze(0).to(predict_config["device"])

        with torch.no_grad():
            pred_dist = model(tensor).cpu().numpy()[0]

        scores = np.arange(1, 11)
        pred_score = np.sum(pred_dist * scores)
        return pred_score
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


# 主预测流程
def run_prediction():
    model = load_model()
    train_set, test_set = load_split_files()
    df = pd.read_excel(predict_config["excel_path"], sheet_name=predict_config["activity"], usecols=[1, 4, 5], skiprows=1, names=['filename', 'true_score', 'comparison_time'])

    results = []

    for idx, row in tqdm(df.iterrows()):
        file_id = int(row['filename'])
        img_path = os.path.join(predict_config["image_dir"],
                                predict_config["activity"],
                                f"{file_id}.jpg")

        if file_id in train_set:
            split = 'train'
        elif file_id in test_set:
            split = 'test'
        else:
            print(f"Warning: {file_id} not in any split, skipping")
            continue

        pred_score = predict_image(model, img_path)

        if pred_score is not None:
            results.append({
                'filename': file_id,
                'split': split,
                'comparison_time': row['comparison_time'],
                'true_score': row['true_score'],
                'pred_score': pred_score / 10
            })

    result_df = pd.DataFrame(results)

    result_df['true_rank'] = result_df.groupby('split')['true_score'].rank(method='average', ascending=False)
    result_df['pred_rank'] = result_df.groupby('split')['pred_score'].rank(method='average', ascending=False)

    test_df = result_df[result_df['split'] == 'test']
    if not test_df.empty:
        corr, p_value = pearsonr(test_df['true_score'], test_df['pred_score'])
        print(f"[Test Set] Pearson Correlation: {corr:.4f} (p={p_value:.4e})")
        shutil.copy(predict_config["model_path"], f'{predict_config["model_path"][:-4]}_{corr:.4f}.pth')
        shutil.copy(train_txt, f'{train_txt[:-4]}_{corr:.4f}.txt')
        shutil.copy(test_txt, f'{test_txt[:-4]}_{corr:.4f}.txt')
    else:
        print("Warning: No test samples found for correlation calculation")

    with pd.ExcelWriter(predict_config["output_excel"], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name=predict_config['activity'], index=False)
    print(f"Results saved to {predict_config['output_excel']}")
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as file:
        file.write(f'Now do the prediction: {formatted_time}\n')
        file.write(f"[Test Set] Pearson Correlation: {corr:.4f} (p={p_value:.4e}).\n\n")

if __name__ == "__main__":
    run_prediction()
