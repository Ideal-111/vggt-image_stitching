import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def get_overlapping_regions(img1, img2, H):
    """
    get overlapping regions from img2 to img 1 coordinates
    H: 
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1_gray = to_gray_uint8(img1)
    img2_gray = to_gray_uint8(img2)
    
    mask1 = get_valid_mask(img1_gray)
    mask2 = get_valid_mask(img2_gray)
    warped_mask2 = cv2.warpPerspective(
        mask2, H, (w1, h1), 
        flags=cv2.INTER_NEAREST
    )

    overlap_mask = cv2.bitwise_and(mask1, warped_mask2)
    warped_img2 = cv2.warpPerspective(
        img2_gray, H, (w1, h1), 
        flags=cv2.INTER_LINEAR
    )
    
    img1_pixels = img1_gray[overlap_mask > 0].flatten()
    img2_pixels = warped_img2[overlap_mask > 0].flatten()

    ssim_score = ssim(img1_gray, warped_img2, data_range=255, full=False)
    print(f"SSIM: {ssim_score:.4f}")
    
    return img1_pixels, img2_pixels, overlap_mask

def calculate_mse(img1, img2):
    """calculate mse"""
    mse = np.mean((img1 - img2) **2)
    return mse

def calculate_psnr(img1, img2, max_pixel=255.0):
    """calculate PSNR"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def to_gray_uint8(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if gray.dtype == np.float32 or gray.dtype == np.float64:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def get_valid_mask(gray_img, threshold=1):
        _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)

def apply_homography(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a homography to a point.

    Args:
        H: Homography matrix
        point: Point to apply the homography to, with shape (2,1)

    Returns:
        new_point: Point after applying the homography, with shape (2,1)
    """
    point = np.asarray([[point[0][0], point[1][0], 1]]).T
    new_point = H @ point
    return new_point[0:2] / new_point[2]


def apply_homography_list(H: np.ndarray, points: list[np.ndarray]) -> list[np.ndarray]:
    """
    Apply a homography to a list of points.

    Args:
        H: Homography matrix
        points: List of points to apply the homography to, each with shape (2,1)

    Returns:
        new_points: List of points after applying the homography, each with shape (2,1)
    """
    return [apply_homography(H, point) for point in points]

def get_new_corners(image: np.ndarray, H: np.ndarray) -> list[np.ndarray]:
    """
    Get the new corners of an image after applying a homography.

    Args:
        image: Original image
        H: Homography matrix

    Returns:
        corners: Corners of the image after applying the homography
    """
    top_left = np.asarray([[0, 0]]).T
    top_right = np.asarray([[image.shape[1], 0]]).T
    bottom_left = np.asarray([[0, image.shape[0]]]).T
    bottom_right = np.asarray([[image.shape[1], image.shape[0]]]).T

    return apply_homography_list(H, [top_left, top_right, bottom_left, bottom_right])


def get_offset(corners: list[np.ndarray]) -> np.ndarray:
    """
    Get offset matrix so that all corners are in positive coordinates.

    Args:
        corners: List of corners of the image

    Returns:
        offset: Offset matrix
    """
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float32,
    )


def get_new_size(corners_images: list[list[np.ndarray]]) -> tuple[int, int]:
    """
    Get the size of the image that would contain all the given corners.

    Args:
        corners_images: List of corners of the images
            (i.e. corners_images[i] is the list of corners of image i)

    Returns:
        (width, height): Size of the image
    """
    top_right_x = np.max([corners_image[1][0] for corners_image in corners_images])
    bottom_right_x = np.max([corners_images[3][0] for corners_images in corners_images])

    bottom_left_y = np.max([corners_images[2][1] for corners_images in corners_images])
    bottom_right_y = np.max([corners_images[3][1] for corners_images in corners_images])

    width = int(np.ceil(max(bottom_right_x, top_right_x)))
    height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

    width = min(width, 5000)
    height = min(height, 4000)

    return width, height

def get_new_parameters(
    panorama: np.ndarray, image: np.ndarray, H: np.ndarray
) -> tuple[tuple[int, int], np.ndarray]:
    """
    Get the new size of the image and the offset matrix.

    Args:
        panorama: Current panorama
        image: Image to add to the panorama
        H: Homography matrix for the image

    Returns:
        size, offset: Size of the new image and offset matrix.
    """
    corners = get_new_corners(image, H)
    added_offset = get_offset(corners)

    corners_image = get_new_corners(image, added_offset @ H)
    if panorama is None:
        size = get_new_size([corners_image])
    else:
        corners_panorama = get_new_corners(panorama, added_offset)
        size = get_new_size([corners_image, corners_panorama])

    return size, added_offset
    