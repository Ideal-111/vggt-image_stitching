import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 

def visualize_attention_map(
    image_tensor:torch.Tensor,
    attn_map_1d_for_patches: torch.Tensor,
    patch_size: int = 14,
    alpha: float = 0.5,
    cmap_name: str = 'viridis',
    output_path: str = None
):
    """
    可视化 Vision Transformer 模型的注意力图。
    该函数将注意力图重塑、上采样并叠加到原始图像上，
    生成类似于热力图的可视化效果。

    Args:
        attn_map_tensor (torch.Tensor): 从模型获取的注意力图张量。
                                        通常是 [CLS] token 对所有 patch token 的注意力，
                                        形状可能为 [num_patches] 或 [1, num_patches]，
                                        或者如果包含 CLS token 则是 [num_patches + 1]。
                                        如果包含批次或多头信息，请确保在传入前已正确提取或平均。
        patch_size (int, optional): Vision Transformer 模型使用的图像块大小。默认为 16。
        alpha (float, optional): 叠加注意力图时的透明度，介于 0.0（完全透明）和 1.0（完全不透明）之间。
                                 默认为 0.6。
        cmap_name (str, optional): Matplotlib 颜色映射的名称，用于生成热力图。
                                   例如 'viridis', 'jet', 'hot', 'plasma', 'inferno'。
                                   默认为 'viridis'。
        output_path (str, optional): 如果提供，将可视化结果保存到指定路径。
                                     如果为 None，则只显示图像。默认为 None。
    """
    original_image = image_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)

    H, W = image_tensor.shape[-2:]

    attn_map_np = attn_map_1d_for_patches.detach().cpu().numpy()

    expected_num_patches = (H // patch_size) * (W // patch_size)
   
    if attn_map_np.size != expected_num_patches:
        raise ValueError(
            f"传入的 attn_map_1d_for_patches 大小 ({attn_map_np.size}) "
            f"与预期图像块网格大小 ({expected_num_patches}) 不匹配。\n"
            f"请确保您已正确提取并移除了 CLS 和 Register token 的注意力。"
        )

    # 2. 将注意力图重塑为 2D 网格
    attn_map_2d = attn_map_np.reshape((H // patch_size, W // patch_size))
    # attn_map_2d = attn_map_np

    # 3. 将注意力图上采样到原始图像尺寸
    # 使用 cv2.resize 进行双三次插值 (bicubic interpolation)
    # upsampled_attn_map = cv2.resize(attn_map_2d, (W, H), interpolation=cv2.INTER_CUBIC)

    attn_map_2d = torch.from_numpy(attn_map_2d).unsqueeze(0).unsqueeze(0)  # 增加批次和通道维度
    upsampled_attn_map = torch.nn.functional.interpolate(
        attn_map_2d, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    
    # 4. 将注意力图归一化到 [0, 1] 范围
    upsampled_attn_map = (upsampled_attn_map - upsampled_attn_map.min()) / \
                         (upsampled_attn_map.max() - upsampled_attn_map.min() + 1e-8)


    cmap = plt.get_cmap(cmap_name)
    colored_attn_map = cmap(upsampled_attn_map)[:, :, :3]

    original_image_float = original_image.astype(np.float32) / 255.0
    overlayed_image = (original_image_float * (1 - alpha) + colored_attn_map * alpha) * 255
    overlayed_image = overlayed_image.astype(np.uint8)


    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("original image")
    axs[0].axis('off')

    axs[1].imshow(original_image)
    im = axs[1].imshow(upsampled_attn_map, cmap=cmap, alpha=alpha)
    axs[1].set_title("Attention map")
    axs[1].axis('off')

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image)
    # plt.title("Original")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(upsampled_attn_map, cmap=cmap_name)
    # plt.title("Heatmap")
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(overlayed_image)
    # plt.title(f"Overlayed, alpha={alpha}")
    # plt.axis('off')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"可视化结果已保存到: {output_path}")
    else:
        plt.show()