import cv2
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from vggt.utils.metrics import get_overlapping_regions, calculate_mse, calculate_psnr, get_new_parameters

class Image:
    def __init__(self, image: np.ndarray, homography: np.ndarray):
        self.image = image
        self.H = homography

def stitch_images(full_path_image_names, tracked_pts, processed_images_for_cv2):

    ref_image = processed_images_for_cv2[0]
    ref_points = tracked_pts[0]

    H_list = []
    for i in range(len(full_path_image_names)):
        current_points = tracked_pts[i]
        H, num = cv2.findHomography(current_points, ref_points, cv2.RANSAC, ransacReprojThreshold=1.0)
        H_list.append(H)

    print(f"✅match points: {np.sum(num)}")


    mi_values = []
    for i in range(1, len(full_path_image_names)):
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
        print(f"mutual information (image {i} with reference image {0}): {mi:.4f}")

        hist1, _ = np.histogram(img_pixels, bins=256, range=(0, np.max(img_pixels)), density=True)
        hist2, _ = np.histogram(ref_pixels, bins=256, range=(0, np.max(ref_pixels)), density=True)
        
        ent1 = entropy(hist1, base=2)
        ent2 = entropy(hist2, base=2)
        CE = ent1 + ent2 - mi
        print(f"joint entropy: {CE:.4f}")

        mse = calculate_mse(img_pixels, ref_pixels)
        psnr = calculate_psnr(img_pixels, ref_pixels)
        print(f"MSE: {mse:.4f}")
        print(f"PSNR: {psnr:.4f} dB")

    image_objects = []
    for img, H in zip(processed_images_for_cv2, H_list):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        image_objects.append(Image(img, H))


    num_bands = 5 
    sigma = 1.0
    stitched_image = multi_band_blending(image_objects, num_bands, sigma)


    return np.clip(stitched_image, 0, 255).astype(np.uint8)


def single_weights_array(size: int) -> np.ndarray:
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])

def single_weights_matrix(shape: tuple[int]) -> np.ndarray:
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    )

def add_weights(weights_matrix: np.ndarray, image: Image, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = offset @ image.H
    size, added_offset = get_new_parameters(weights_matrix, image.image, H)

    weights = single_weights_matrix(image.image.shape)
    weights = cv2.warpPerspective(weights, added_offset @ H, size)[:, :, np.newaxis]

    if weights_matrix is None:
        weights_matrix = weights
    else:
        weights_matrix = cv2.warpPerspective(weights_matrix, added_offset, size)
        if len(weights_matrix.shape) == 2:
            weights_matrix = weights_matrix[:, :, np.newaxis]
        weights_matrix = np.concatenate([weights_matrix, weights], axis=2)

    return weights_matrix, added_offset @ offset

def get_max_weights_matrix(images: list[Image]) -> tuple[np.ndarray, np.ndarray]:
    weights_matrix = None
    offset = np.eye(3)
    for image in images:
        weights_matrix, offset = add_weights(weights_matrix, image, offset)

    weights_maxes = np.max(weights_matrix, axis=2)[:, :, np.newaxis]
    max_weights_matrix = np.where(
        np.logical_and(weights_matrix == weights_maxes, weights_matrix > 0), 1.0, 0.0
    )
    return np.transpose(max_weights_matrix, (2, 0, 1)), offset

def get_cropped_weights(images: list[Image], weights: np.ndarray, offset: np.ndarray) -> list[np.ndarray]:
    cropped_weights = []
    for i, image in enumerate(images):
        cropped_weights.append(
            cv2.warpPerspective(
                weights[i], np.linalg.inv(offset @ image.H), image.image.shape[:2][::-1]
            )
        )
    return cropped_weights

def build_band_panorama(images: list[Image], weights: list[np.ndarray], bands: list[np.ndarray], offset: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pano_weights = np.zeros(size)
    pano_bands = np.zeros((*size, 3))

    for i, image in enumerate(images):
        weights_at_scale = cv2.warpPerspective(weights[i], offset @ image.H, size[::-1])
        pano_weights += weights_at_scale
        pano_bands += weights_at_scale[:, :, np.newaxis] * cv2.warpPerspective(
            bands[i], offset @ image.H, size[::-1]
        )

    return np.divide(
        pano_bands, pano_weights[:, :, np.newaxis], where=pano_weights[:, :, np.newaxis] != 0
    )

def multi_band_blending(images: list[Image], num_bands: int, sigma: float) -> np.ndarray:
    max_weights_matrix, offset = get_max_weights_matrix(images)
    size = max_weights_matrix.shape[1:]

    max_weights = get_cropped_weights(images, max_weights_matrix, offset)

    weights = [[cv2.GaussianBlur(max_weights[i], (0, 0), 2 * sigma) for i in range(len(images))]]
    sigma_images = [cv2.GaussianBlur(image.image, (0, 0), sigma) for image in images]
    bands = [
        [
            np.where(
                images[i].image.astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                images[i].image - sigma_images[i],
                0,
            )
            for i in range(len(images))
        ]
    ]

    for k in range(1, num_bands - 1):
        sigma_k = np.sqrt(2 * k + 1) * sigma
        weights.append(
            [cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))]
        )

        old_sigma_images = sigma_images
        sigma_images = [
            cv2.GaussianBlur(old_sigma_image, (0, 0), sigma_k)
            for old_sigma_image in old_sigma_images
        ]
        bands.append(
            [
                np.where(
                    old_sigma_images[i].astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                    old_sigma_images[i] - sigma_images[i],
                    0,
                )
                for i in range(len(images))
            ]
        )

    # the last band
    sigma_k = np.sqrt(2 * (num_bands - 1) + 1) * sigma if num_bands > 1 else sigma
    weights.append([cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))])
    bands.append([sigma_images[i] for i in range(len(images))])

    panorama = np.zeros((*max_weights_matrix.shape[1:], 3))
    for k in range(num_bands):
        panorama += build_band_panorama(images, weights[k], bands[k], offset, size)
        panorama = np.clip(panorama, 0, 255)

    return panorama