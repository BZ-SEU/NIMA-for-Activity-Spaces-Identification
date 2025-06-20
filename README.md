# Urban Informal Fitness Activity Potential Assessment Project

This project leverages the **NIMA (Neural Image Assessment)** model paradigm to evaluate urban spaces and generate potential scores for informal fitness activities. The results provide scientific guidance for citizens' daily exercise routines and urban facility planning.

## Project Background

**Informal fitness activities** refer to spontaneous physical exercises conducted by residents, such as:
1. Square dancing activities
2. Linear activities (walking/running)
3. Ball sports (badminton/basketball/football/volleyball)
4. Roller skating/skateboarding
5. Children's activities
6. Individual exercises (e.g., Tai Chi, equipment workouts)

Typical spaces for these activities include:
- Parks and gardens
- Green facilities
- Natural/semi-natural open spaces
- Urban squares
- Street open areas

## Dataset Description
The project includes a dedicated dataset collected by our research team:
- **Data Structure**:
  - Spatial images of venues
  - Activity occurrence potential labels (generated using TrueSkill™ algorithm)
- **Initial Activity Types** (10 categories):
  1. Badminton
  2. Basketball
  3. Children's activities
  4. Football
  5. Individual exercises
  6. Linear activities (walking/running)
  7. Square dancing
  8. Roller skating
  9. Skateboarding
  10. Volleyball
- **Expandability**: Supports adding custom datasets for new activities

## Usage Guide

### 1. Dataset Splitting
```bash
python split_dataset.py --chunk [activity_name]
# Example (football):
python split_dataset.py --chunk football
```

### 2. Model Training
```bash
python train.py --chunk [activity_name]
# Example (football):
python train.py --chunk football
```

### 3. Potential Score Prediction
```bash
python predict.py --chunk [activity_name]
# Example (football):
python predict.py --chunk football
```

### 4. Model Analysis (Grad-CAM Visualization)
```bash
python grad_cam_batch.py --chunk [activity_name]
# Example (football):
python grad_cam_batch.py --chunk football
```

## Technical Highlights
- **NIMA Model**: Assesses aesthetic quality of images adapted for urban space potential scoring
- **Grad-CAM**: Gradient-weighted Class Activation Mapping for model behavior analysis
- **TrueSkill™ Algorithm**: Generates reliable activity potential assessment labels

---

# 城市非正规健身活动潜力评估项目

本项目利用图像质量评估模型 **NIMA (Neural Image Assessment)** 范式，对城市空间进行评估，获取其发生非正规健身活动的潜力评分，从而为市民日常锻炼与城市设施建设提供科学建议。

## 项目背景

**非正规健身活动**指居民自发进行的身体锻炼活动形式，例如：
1. 广场舞活动
2. 散步/跑步等线性运动
3. 球类活动（羽毛球/篮球/足球/排球）
4. 轮滑/滑板
5. 儿童活动
6. 个体运动（如太极拳、器械锻炼等）

此类活动的典型承载空间包括：
- 公园和花园
- 设施绿地
- 自然/半自然开放空间
- 城市广场
- 街道空地

## 数据集说明
包含由本项目组收集的专用数据集：
- **数据构成**：
  - 场地空间图像
  - 活动发生潜力评估标签（通过Trueskill算法生成）
- **初始活动类型**（10类）：
  1. 羽毛球  
  2. 篮球  
  3. 儿童活动  
  4. 足球  
  5. 个体运动  
  6. 线性运动（散步/跑步等）  
  7. 广场舞  
  8. 轮滑  
  9. 滑板  
  10. 排球
- **扩展性**：支持自行添加其他活动数据集

## 使用指南

### 1. 数据集划分
```bash
python split_dataset.py --chunk [活动名称]
# 示例（足球）：
python split_dataset.py --chunk football
```

### 2. 模型训练
```bash
python train.py --chunk [活动名称]
# 示例（足球）：
python train.py --chunk football
```

### 3. 潜力评分预测
```bash
python predict.py --chunk [活动名称]
# 示例（足球）：
python predict.py --chunk football
```

### 4. 模型分析（Grad-CAM可视化）
```bash
python grad_cam_batch.py --chunk [活动名称]
# 示例（足球）：
python grad_cam_batch.py --chunk football
```

## 技术亮点
- **NIMA模型**：评估图像美学质量，适配城市空间潜力评分
- **Grad CAM**：梯度加权类激活映射技术解析模型关注区域
- **TrueSkill™算法**：生成可靠的活动潜力评估标签
