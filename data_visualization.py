import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv("processed_audio_features.csv")

# 检查数据基本信息
print(data.info())
print(data.head())

# 检查特征分布
mfcc_columns = [col for col in data.columns if "mfcc" in col]
data[mfcc_columns].hist(bins=30, figsize=(12, 10))
plt.suptitle("MFCC Features Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# 热力图展示特征相关性
chroma_columns = [col for col in data.columns if "chroma" in col]
corr_matrix = data[mfcc_columns + chroma_columns + ["zcr", "spectral_centroid"]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Matrix", fontsize=16)
plt.show()
