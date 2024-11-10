import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_metrics(output_file, label_file='label/train(1).txt'):
    # 读取output文件和label文件
    output_df = pd.read_csv(output_file)
    label_df = pd.read_csv(label_file)

    # 通过guid匹配数据
    merged_df = pd.merge(output_df, label_df, on='guid')

    # 计算准确率和F1分数
    accuracy = accuracy_score(merged_df['tag_x'], merged_df['tag_y'])
    f1 = f1_score(merged_df['tag_x'], merged_df['tag_y'], average='weighted')

    # 获取标签（假设标签是数字或类别名称）
    display_labels = sorted(merged_df['tag_y'].unique())

    # 生成并显示混淆矩阵图
    cm = confusion_matrix(merged_df['tag_y'], merged_df['tag_x'], labels=display_labels)

    # 创建混淆矩阵显示对象，并指定颜色映射为蓝色系
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    # 设置图像大小和分辨率
    plt.figure(figsize=(10, 10), dpi=300)
    
     
    disp.plot(cmap='Blues', text_kw={'fontsize': 11})  # 调整数字大小
    plt.title('Confusion Matrix - Test')
    plt.show()

    return accuracy, f1

# 使用函数
output_file = 'output/test_with_label.txt'
accuracy, f1 = calculate_metrics(output_file)
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
