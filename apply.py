# 作   者:BZ
# 开发时间:2025/3/10
import os

import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from NIMA_1 import NIMA
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chunk', default='basketball', help='chunk')
args = parser.parse_args()

# 配置参数
predict_config = {
    "model_path": f"./trained_models/nima_{args.chunk}.pth",
    "image_dir": r"E:\StreetView",
    "activity": args.chunk,
    "output_excel": f"./apply_results.xlsx",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

predict_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载模型
def load_model():
    model = NIMA()
    model.load_state_dict(torch.load(predict_config["model_path"]))
    model.eval()
    return model.to(predict_config["device"])


# 单张图片预测函数
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


def run_application():
    model = load_model()

    files = os.listdir(predict_config["image_dir"])
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]

    results = []

    for file in tqdm(jpg_files):
        img_path = os.path.join(predict_config["image_dir"], file)

        pred_score = predict_image(model, img_path)

        if pred_score is not None:
            results.append({
                'filename': file,
                'activity': predict_config["activity"],
                'pred_score': pred_score / 10
            })

    result_df = pd.DataFrame(results)

    with pd.ExcelWriter(predict_config["output_excel"], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name=predict_config['activity'], index=False)
    print(f"Results saved to {predict_config['output_excel']}")

if __name__ == "__main__":
    run_application()
