import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def plot_tsne(root, epoch, name):
    # 加载三组数据
    InD_train_feature = np.load(f'{root}/epoch_{epoch}_InD_train_feature.npy')
    InD_test_feature = np.load(f'{root}/epoch_{epoch}_InD_test_feature.npy')
    OOD_tiny_imagenet_feature = np.load(f'{root}/epoch_{epoch}_OOD_{name}_feature.npy')
    # 随机抽取每组数据的200个样本
    sample_size = 24
    sample_indices = np.random.choice(InD_train_feature.shape[0], sample_size, replace=False)
    InD_train_feature_ = InD_train_feature[sample_indices]
    sample_indices = np.random.choice(InD_test_feature.shape[0], sample_size, replace=False)
    InD_test_feature_ = InD_test_feature[sample_indices]
    sample_indices = np.random.choice(OOD_tiny_imagenet_feature.shape[0], sample_size, replace=False)
    OOD_tiny_imagenet_feature_ = OOD_tiny_imagenet_feature[sample_indices]

    # 合并抽样数据和标签
    combined_data = np.vstack((InD_train_feature_, InD_test_feature_, OOD_tiny_imagenet_feature_))
    labels = np.repeat(['InD Train', 'InD Test', 'OOD Test'], sample_size)

    # 为每个数据组设置颜色
    colors = ['blue', 'orange', 'green']

    # 创建t-SNE对象
    tsne = TSNE(n_components=2, random_state=42)

    # 对合并数据进行降维
    data_tsne = tsne.fit_transform(combined_data)

    # 绘制t-SNE图，使用不同颜色表示不同数据组
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(data_tsne[labels == label, 0],
                    data_tsne[labels == label, 1], color=colors[i], label=label, marker='.', s=50, alpha=0.7)    # s 大小

    plt.title('t-SNE Visualization of Three Data Groups')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 添加图例
    plt.legend(loc='upper right')

    # 创建目录（文件夹）并保存t-SNE图
    output_directory = f'{root}/tsne/'
    os.makedirs(output_directory, exist_ok=True)
    plt.savefig(os.path.join(output_directory, f'epoch_{epoch}_tsne_plot_{name}.png'), dpi=300)
