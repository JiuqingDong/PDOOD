import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F
from src.utils.file_io import PathManager

def plot_distribution(id_scores, ood_scores, epoch, method, output_path = None):
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    plt.savefig(os.path.join(f"{output_path}/{method}_epoch_{epoch}.png"), bbox_inches='tight')

def plot_distribution_1(id_scores, ood_scores, epoch, method, output_path=None):
    min_value = np.minimum(id_scores.min(), ood_scores.min())
    max_value = np.maximum(id_scores.max(), ood_scores.max())
    gap = max_value - min_value
    # 将数据归一化
    id_scores_normalized = (id_scores - min_value) / gap
    ood_scores_normalized = (ood_scores - min_value) / gap

    # 设置样式和调色板
    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']

    # 绘制归一化的分布图
    sns.displot({"ID": id_scores_normalized, "OOD": ood_scores_normalized}, label="id", kind="kde", palette=palette, fill=True, alpha=0.8)
    # sns.histplot({"ID": id_scores_normalized, "OOD": ood_scores_normalized}, kde = True, alpha=0.8, bins = 100)
    plt.xlim(-0.1, 1.1)

    # 保存图像
    if epoch == 99:
        plt.savefig(os.path.join(f"{output_path}/{method}_last.png"), bbox_inches='tight', dpi=200)
    else:
        plt.savefig(os.path.join(f"{output_path}/epoch_{epoch}_{method}.png"), bbox_inches='tight', dpi=200)


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9)
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


