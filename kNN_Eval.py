import numpy as np
from tsne import plot_tsne


def kNN_OOD(root, epoch, name):
    ''' implementation of KNN '''

    InD_train_dataset_path = f"{root}/epoch_{epoch}_InD_train_feature.npy"  # 替换为你的实际文件路径
    InD_test_dataset_path = f"{root}/epoch_{epoch}_InD_test_feature.npy"  # 替换为你的实际文件路径
    OOD_dataset_path = f"{root}/epoch_{epoch}_OOD_{name}_feature.npy"  # 替换为你的实际文件路径

    # 使用np.load()函数读取npy文件
    InD_train_dataset = np.load(InD_train_dataset_path)
    InD_test_dataset = np.load(InD_test_dataset_path)
    OOD_dataset = np.load(OOD_dataset_path)

    InD_similarity = cosine_similarity_matrix(InD_test_dataset, InD_train_dataset)
    OOD_similarity = cosine_similarity_matrix(OOD_dataset, InD_train_dataset)

    InD_logits = 1 - np.max(InD_similarity, axis=1)
    OOD_logits = 1 - np.max(OOD_similarity, axis=1)


    plot_tsne(root, epoch, name)

    return InD_logits, OOD_logits


def cosine_similarity_matrix(matrix1, matrix2):
    # 计算矩阵每行的范数（欧几里德范数）
    norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)

    # 将矩阵标准化为单位向量
    normalized_matrix1 = matrix1 / norm_matrix1
    normalized_matrix2 = matrix2 / norm_matrix2

    # 计算相似度矩阵，即标准化后的测试样本与训练样本之间的余弦相似度
    similarity_matrix = np.dot(normalized_matrix1, normalized_matrix2.T)
    # similarity_matrix = np.dot(matrix1, matrix2.T)

    return similarity_matrix
