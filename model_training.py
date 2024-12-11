import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 读取已处理的特征数据
DATA_PATH = "processed_audio_features.csv"
data = pd.read_csv(DATA_PATH)

# 编码目标变量
label_encoder = LabelEncoder()
data["Story_type"] = label_encoder.fit_transform(data["Story_type"])

# 分离特征和目标变量
X = data.drop(columns=["filename", "Story_type", "Language"])
y = data["Story_type"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# 测试集预测
y_pred = rf_model.predict(X_test)

# 模型性能评估
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(conf_matrix)

# 分类报告
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\n分类报告:")
print(class_report)

# 混淆矩阵可视化
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(conf_matrix, label_encoder.classes_)

# 分类报告可视化
def plot_classification_report(report, labels):
    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="coolwarm", fmt=".2f", cbar=True, xticklabels=report_df.columns[:-1], yticklabels=report_df.index[:-1])
    plt.title("Classification Report", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Classes", fontsize=12)
    plt.tight_layout()
    plt.show()

plot_classification_report(class_report, label_encoder.classes_)

# 特征重要性可视化
def plot_feature_importance(model, features):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_r")
    plt.title("Feature Importance", fontsize=16)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_model, X.columns)
