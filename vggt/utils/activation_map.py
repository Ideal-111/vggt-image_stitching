import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_activation_map(aggregated_tokens, image_path, image_tensor, patch_size, ps_idx=0, layer_idx=-1, alpha=0.5, cmap='jet'):
    """
    可视化 transformer 模型中某层输出的激活图。

    参数：
    - aggregated_tokens: List of tensors，模型输出的 token 序列（如 [layer1_output, layer2_output, ...]）
                         每个元素形状为 [B, S, P_total, C]
    - image_path: str，原始图像路径（用于显示原图）
    - image_tensor: torch.Tensor，输入图像张量（形状为 [B, C, H, W]），用于获取原始分辨率
    - patch_size: int，patch 的尺寸（如 ViT 为 16）
    - ps_idx: int，保留 patch token 的起始索引（默认 0）
    - layer_idx: int，选择可视化哪一层的 token（默认为最后一层）
    - alpha: float，热力图叠加透明度（默认 0.5）
    - cmap: str，colormap 类型（默认 'jet'）
    """

    chosen_layer_output = aggregated_tokens[layer_idx]  # shape: [B, S, P_total, C]
    B, S, P_total, C = chosen_layer_output.shape

    patch_tokens_only = chosen_layer_output[0, 0, ps_idx:, :]  # shape: [N, C]

    original_H, original_W = image_tensor.shape[-2:]
    grid_H, grid_W = original_H // patch_size, original_W // patch_size

    spatial_features = patch_tokens_only.view(grid_H, grid_W, C)
    activation_map = spatial_features.sum(dim=-1)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())

    activation_map_upsampled = F.interpolate(
        activation_map.unsqueeze(0).unsqueeze(0),
        size=(original_H, original_W),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    original_image = Image.open(image_path)
    original_image_display = np.array(original_image)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(original_image_display)
    axs[0].set_title("original image")
    axs[0].axis('off')

    axs[1].imshow(original_image_display)
    im = axs[1].imshow(activation_map_upsampled, cmap=cmap, alpha=alpha)
    axs[1].set_title("activation map")
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig('activation_visualization.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()