# 作   者:BZ
# 开发时间:2025/3/4
import os
from random import shuffle
from datetime import datetime
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chunk', default='basketball', help='chunk')
args = parser.parse_args()

split_now = args.chunk
split_ratio = 0.8

image_dir = 'All_images/' + split_now + '/'
label_dir = 'All_labels/Result0310.xlsx'
output_train = 'dataset_split/' + split_now + '_train_files.txt'
output_test = 'dataset_split/' + split_now + '_test_files.txt'

image_files = {f[:-4] for f in os.listdir(image_dir) if f.endswith('.jpg')}

file_list = list(image_files)

shuffle(file_list)

split_point = int(len(file_list) * split_ratio)

train_list = file_list[:split_point]
test_list = file_list[split_point:]

with open(output_train, 'w') as f:
    for file_name in train_list:
        f.write(f"{file_name}\n")

with open(output_test, 'w') as f:
    for file_name in test_list:
        f.write(f"{file_name}\n")

current_time = datetime.now()
log_file = 'logs/' + split_now + '_log.txt'
with open(log_file, 'a') as file:
    file.write(f"Split the dataset: ")
    file.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    file.write(f'Number of train set: {len(train_list)}\n')
    file.write(f'Number of test set: {len(test_list)}\n\n')
