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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置参数
config = {
    "model_cache": {},
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "modelpath_dir": r'trained_models/',
    "input_excel": r'application_file/20250323judge_result.xlsx',
    "image_dir": r'F:\StreetView\StreetView_other',
    "output_excel": r'application_file/20250323NIMA_result.xlsx'
}

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_path):
    if model_path not in config["model_cache"]:
        model = NIMA()
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            raise ValueError(f"无法加载模型权重：{model_path}")
        model.eval()
        model = model.to(config["device"])
        config["model_cache"][model_path] = model
    return config["model_cache"][model_path]


def predict_image(model, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        tensor = tensor.to(config["device"])

        with torch.no_grad():
            pred_dist = model(tensor).cpu().numpy()[0]

        image.close()
        scores = np.arange(1, 11)
        pred_score = np.sum(pred_dist * scores) / 10  # 直接标准化到0.1~1.0
        return pred_score
    except Exception as e:
        print(f"处理失败：{image_path} - {str(e)}")
        return 0.0  # 错误时返回0


def process_excel():
    df = pd.read_excel(config['input_excel'])
    model_columns = df.columns[1:]

    for model_col in model_columns:
        model_path = config['modelpath_dir'] + 'nima_' + model_col + '.pth'
        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"跳过无效模型列：{model_col} - {str(e)}")
            continue

        for idx in tqdm(df.index, desc=f"处理{os.path.basename(model_col)}"):
            if df.at[idx, model_col] == 1:
                filename = df.at[idx, df.columns[0]]
                img_path = os.path.join(config['image_dir'], filename)
                df.at[idx, model_col] = predict_image(model, img_path)
            else:
                df.at[idx, model_col] = 0.0

        df.to_excel(config['output_excel'], index=False)
        print(f"{model_col}结果已保存至：{config['output_excel']}")


if __name__ == "__main__":
    process_excel()
