import os
import pandas as pd
import librosa
import numpy as np

# 定义文件路径
CSV_FILE = "CBU0521DD_stories_attributes.csv"  # CSV 文件路径
AUDIO_FOLDER = "CBU0521DD_stories"  # 音频文件夹路径
OUTPUT_FILE = "processed_audio_features.csv"  # 输出文件路径

# 定义音频特征提取函数
def extract_audio_features(file_path):
    """
    提取音频文件的特征，包括 MFCC, Chroma, ZCR 和谱质心。
    :param file_path: 音频文件的路径
    :return: 提取的特征数组
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)  # 加载音频，采样率16kHz
        # 提取 MFCC 特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        # 提取 Chroma 特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        # 提取 ZCR（零交叉率）
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        # 提取谱质心（Spectral Centroid）
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        # 合并特征
        return np.concatenate([mfcc, chroma, [zcr, spectral_centroid]])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 加载 CSV 文件
try:
    metadata = pd.read_csv(CSV_FILE)
    print(f"Loaded metadata from {CSV_FILE}")
    print("Columns in CSV file:", metadata.columns.tolist())  # 打印列名验证
except FileNotFoundError:
    print(f"CSV file not found at {CSV_FILE}. Please check the path.")
    exit()

# 修改为实际的列名 "filename"
metadata["file_id"] = metadata["filename"].str.split(".").str[0]  # 提取文件编号

# 初始化特征列表
feature_list = []
file_ids = []

# 遍历音频文件夹中的文件
if not os.path.exists(AUDIO_FOLDER):
    print(f"Audio folder not found at {AUDIO_FOLDER}. Please check the path.")
    exit()

for file_name in os.listdir(AUDIO_FOLDER):
    if file_name.endswith(".wav"):
        file_path = os.path.join(AUDIO_FOLDER, file_name)
        print(f"Processing file: {file_name}")
        features = extract_audio_features(file_path)
        if features is not None:
            feature_list.append(features)
            file_ids.append(file_name.split(".")[0])  # 记录文件编号
        else:
            print(f"Failed to extract features from {file_name}")

# 检查是否成功提取了任何特征
if not feature_list:
    print("No features were extracted. Please check your audio files.")
    exit()

# 将提取的特征转换为 DataFrame
feature_columns = [f"mfcc_{i+1}" for i in range(13)] + \
                  [f"chroma_{i+1}" for i in range(12)] + \
                  ["zcr", "spectral_centroid"]
features_df = pd.DataFrame(feature_list, columns=feature_columns)
features_df["file_id"] = file_ids

# 合并特征和元数据
processed_data = pd.merge(metadata, features_df, on="file_id", how="inner")

# 保存处理后的数据
try:
    processed_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Processed data saved to {os.path.abspath(OUTPUT_FILE)}")
except Exception as e:
    print(f"Failed to save processed data: {e}")
