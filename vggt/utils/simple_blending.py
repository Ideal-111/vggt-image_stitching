import numpy as np
import cv2
from vggt.utils.metrics import get_overlapping_regions, calculate_mse, calculate_psnr
from sklearn.metrics.cluster import mutual_info_score
from scipy.stats import entropy
def stitch_images(full_path_image_names, tracked_pts, processed_images_for_cv2):
    """
    Args:
    full_path_image_names: List of full path names of the images.
    tracked_pts: List of tracked feature points for each image.
    processed_images: List of preprocessed images.

    Returns:
    stitched_image: The stitched panorama image (np.ndarray).
    """

    middle_index = len(full_path_image_names) // 2
    ref_image = processed_images_for_cv2[middle_index]
    ref_points = tracked_pts[middle_index]
    ref_h, ref_w = processed_images_for_cv2[middle_index].shape[:2]
    # ref_points = tracked_pts[0]
    # ref_h, ref_w = processed_images_for_cv2[0].shape[:2]

    H_list = []
    for i in range(len(full_path_image_names)):
        current_points = tracked_pts[i]
        H, num = cv2.findHomography(current_points, ref_points, cv2.RANSAC, ransacReprojThreshold=1.0)
        H_list.append(H)

    print(f"✅match points: {np.sum(num)}")

    mi_values = []
    for i in range(len(full_path_image_names)):
        if i == middle_index: 
            # mi_values.append(1.0)
            continue
            
        ref_pixels, img_pixels, _ = get_overlapping_regions(
            ref_image, 
            processed_images_for_cv2[i], 
            H_list[i]
        )
        
        if len(img_pixels) < 2 or len(ref_pixels) < 2:
            mi_values.append(0.0)
            continue
        
        mi = mutual_info_score(img_pixels, ref_pixels)
        mi_values.append(mi)
        print(f"互信息 (图像 {i} 与参考图像 {middle_index}): {mi:.4f}")

        hist1, _ = np.histogram(img_pixels, bins=256, range=(0, np.max(img_pixels)), density=True)
        hist2, _ = np.histogram(ref_pixels, bins=256, range=(0, np.max(ref_pixels)), density=True)
        
        # Compute entropy
        ent1 = entropy(hist1, base=2)
        ent2 = entropy(hist2, base=2)

        CE = ent1 + ent2 - mi
        print(f"联合熵: {CE:.4f}")

        # calculate mse and psnr
        mse = calculate_mse(img_pixels, ref_pixels)
        psnr = calculate_psnr(img_pixels, ref_pixels)

        print(f"MSE: {mse:.4f}")
        print(f"PSNR: {psnr:.4f} dB")

    min_x, min_y = 0, 0
    max_x, max_y = ref_w, ref_h

    for i, H in enumerate(H_list):
        h, w = processed_images_for_cv2[i].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        min_x = int(min(min_x, transformed_corners[:, :, 0].min()) - 1)
        min_y = int(min(min_y, transformed_corners[:, :, 1].min()) - 1)
        max_x = int(max(max_x, transformed_corners[:, :, 0].max()) + 1)
        max_y = int(max(max_y, transformed_corners[:, :, 1].max()) + 1)

    new_width = max_x - min_x
    new_height = max_y - min_y
    offset_x = -min_x
    offset_y = -min_y
    
    stitched_image = np.zeros((new_height, new_width, 3), dtype=np.float32)
    weight_sum = np.zeros((new_height, new_width), dtype=np.float32)

    H_base = H_list[middle_index]
    T_base = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    Homo_base_final = T_base @ H_base

    for i in range(len(full_path_image_names)):
        img = processed_images_for_cv2[i].astype(np.float32)
        H = H_list[i]
        
        h, w = img.shape[:2]
        img_weights = single_weights_matrix((h, w))
        
        img_weights_3ch = np.repeat(
            cv2.warpPerspective(img_weights, T_base @ H, (new_width, new_height))[:, :, np.newaxis],
            3,
            axis=2
        )
        
        warped_img = cv2.warpPerspective(img, T_base @ H, (new_width, new_height))
        
        stitched_image += warped_img * img_weights_3ch
        weight_sum += img_weights_3ch[:, :, 0]

    weight_sum[weight_sum == 0] = 1e-8 
    stitched_image = stitched_image / weight_sum[:, :, np.newaxis]

    stitched_image = np.clip(stitched_image, 0, 255).astype(np.uint8)

    return stitched_image


def single_weights_array(size: int) -> np.ndarray:
    """
    Create a 1D weights array.

    Args:
        size: Size of the array

    Returns:
        weights: 1D weights array
    """
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])

def single_weights_matrix(shape: tuple[int]) -> np.ndarray:
    """
    Create a 2D weights matrix.

    Args:
        shape: Shape of the matrix

    Returns:
        weights: 2D weights matrix
    """
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    )