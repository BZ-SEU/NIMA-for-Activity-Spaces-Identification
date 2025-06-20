# 作   者:BZ
# 开发时间:2025/3/3
import os
from datetime import datetime

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from PIL import Image

from NIMA_1 import *
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chunk', default='plazadancing', help='chunk')
args = parser.parse_args()


# --------------------------------------
# 配置参数
# --------------------------------------
config = {
    "excel_path": "./All_labels/Result0316.xlsx",  # Excel文件路径
    "image_dir": "./All_images",  # 图片文件夹路径
    "activity": args.chunk,
    "batch_size": 16,
    "num_workers": 4,
    "lr": 2e-05,
    "epochs": 40,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": f"./trained_models/nima_{args.chunk}.pth"
}


train_txt = 'dataset_split/' + config['activity'] + '_train_files.txt'
log_file = 'logs/' + config['activity'] + '_log.txt'


class AestheticDataset(Dataset):
    def __init__(self, df, transform=None):
        with open(train_txt, 'r') as f:
            train_files = [line.strip() for line in f]

        train_ids = [int(f) for f in train_files]

        self.df = df[df['filename'].isin(train_ids)].reset_index(drop=True)
        self.transform = transform

        self._create_score_distributions()

    def _create_score_distributions(self):
        self.score_dists = []
        for score in self.df['score']:
            dist = torch.zeros(10)
            idx = int(round(score * 10)) - 1
            idx = max(0, min(idx, 9))
            dist[idx] = 1.0
            self.score_dists.append(dist)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(
            config["image_dir"],
            config["activity"],
            str(int(self.df.iloc[idx]['filename'])) + '.jpg'
        )
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
            image = normalized(image)

        score_dist = self.score_dists[idx]

        return image, score_dist

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def prepare_dataloaders():
    df = pd.read_excel(config["excel_path"], sheet_name=config["activity"], usecols=[1, 4], skiprows=1, names=['filename', 'score'])

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = AestheticDataset(train_df, train_transform)
    val_dataset = AestheticDataset(val_df, train_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    return train_loader, val_loader

def train_model():
    print(f'Now training {args.chunk}.')
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as file:
        file.write(f'Training begin Time: {formatted_time}\n')

    train_loader, val_loader = prepare_dataloaders()
    model = NIMA().to(config["device"])
    criterion = EarthMoverDistanceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999), weight_decay=1e-04)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

    best_val_loss = float('inf')

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(config["device"])
            targets = targets.to(config["device"])

            optimizer.zero_grad()

            outputs = model(images)
            loss = normalized(criterion(outputs, targets))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(config["device"])
                targets = targets.to(config["device"])

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step()

        print(f'Epoch {epoch + 1}/{config["epochs"]} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f}')
        with open(log_file, 'a') as file:
            file.write(f'Epoch {epoch + 1}/{config["epochs"]} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}.\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["save_path"])
        elif pd.isna(val_loss):
            break

    print(f"Saved best model with val loss: {best_val_loss:.4f}")
    with open(log_file, 'a') as file:
        file.write(f"Saved best model with val loss: {best_val_loss:.4f}\n\n")

if __name__ == "__main__":
    train_model()