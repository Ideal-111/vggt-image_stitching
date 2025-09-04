import os
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.utils.keypoint import transform_keypoints
from vggt.utils.stitching import linearBlending
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
# dtype = torch.float16

num_registers = 4
model = VGGT()
state_dict = torch.load("/data/lx/ckpt/model.pt", map_location=device)
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

kpts_path = "/home/lx/ALIKED/output_keypoints/hill_keypoints.pt"
output = "stitched_processed_results/hill.jpg"
input = "examples/hill"


all_files = os.listdir(input)
all_files = sorted(all_files)
full_path_image_names = []
image_names = [file for file in all_files if file.endswith('.jpg')]
for name in image_names:
    full_path = os.path.join(input, name)
    full_path_image_names.append(full_path)

img = cv2.imread(full_path_image_names[0])  # shape: (H, W, 3)
H, W = img.shape[0], img.shape[1]
original_size = (W, H)

images = load_and_preprocess_images(full_path_image_names).to(device)  # shape: [S, 3, H, W]
images_vis = images
  
processed_images_for_cv2 = []
for i in range(images.shape[0]): # Iterate through each image in the batch (S images)
    # Permute dimensions from [C, H, W] to [H, W, C] and move to CPU
    img_current = images[i].permute(1, 2, 0).cpu().numpy()

    # Scale pixel values from [0, 1] to [0, 255] and convert to uint8
    img_current = (img_current * 255).astype(np.uint8)

    # Convert color space from RGB to BGR (as OpenCV expects BGR by default)
    img_current = cv2.cvtColor(img_current, cv2.COLOR_RGB2BGR)
    processed_images_for_cv2.append(img_current)

# inference
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None] 
        print(images.shape)
        aggregated_tokens_list, ps_idx = model.aggregator(images)

        # predict tracks
        # choose your own points to track, with shape (N, 2) for one scene
        original_kpts = torch.load(kpts_path)
        kpts_tensor = torch.load(kpts_path).cpu().numpy()  # shape [N, 2]
        kpts_tensor_numpy = transform_keypoints(kpts_tensor, original_size, target_size=518, mode="crop")
        kpts_tensor_trans = torch.from_numpy(kpts_tensor_numpy)
        query_points = kpts_tensor_trans.to(dtype=torch.float32, device=device)     # shape [N, 2]
  
        track_list, vis_score, conf_score = model.track_head(
            aggregated_tokens_list, 
            images, 
            ps_idx, 
            query_points=query_points[None]
        )                                             # track_list[-1] shape: [B, S, N, 2] 
        tracks = track_list[-1].squeeze(0)            # shape: [S, N, 2]
        
        # use conf_score
        # use vis_score
        conf_score_seq = conf_score.squeeze(0)        # shape: [S, N]
        print(f"max value: {max(conf_score_seq[1])}")
        conf_mask = conf_score_seq >= 0.10          # shape: [N]
        vis_score_seq = vis_score.squeeze(0)        # shape: [S, N]
        print(f"max value: {max(vis_score_seq[1])}")
        vis_mask = vis_score_seq >= 0.40           # shape: [N]
        integral_mask = vis_mask[0] & conf_mask[0] # shape: [N], bool
        
        for i in range(1, len(full_path_image_names)):
            integral_mask = integral_mask & vis_mask[i] & conf_mask[i]

        track_mask = torch.zeros_like(conf_mask, dtype=torch.bool)
        for i in range(len(full_path_image_names)):
            track_mask[i, integral_mask] = True
    
        src_pts = query_points.cpu().numpy()[integral_mask.cpu().numpy()]
        tracked_pts = tracks.cpu().numpy()[:, integral_mask.cpu().numpy()]

        middle_index = len(full_path_image_names) // 2
        ref_points = tracked_pts[middle_index]
        ref_h, ref_w = processed_images_for_cv2[middle_index].shape[:2]
        # ref_points = tracked_pts[0]
        # ref_h, ref_w = processed_images_for_cv2[0].shape[:2]

        H_list = []
        for i in range(len(full_path_image_names)):
            current_points = tracked_pts[i]
            H, num = cv2.findHomography(current_points, ref_points, cv2.RANSAC, ransacReprojThreshold=5.0)
            H_list.append(H)

        print(f"✅成功匹配点个数: {np.sum(num)}")

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

        stitched_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        H_base = H_list[middle_index]
        T_base = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
        Homo_base_final = T_base @ H_base

        base_image = processed_images_for_cv2[middle_index]
        warped_base_image = cv2.warpPerspective(base_image, Homo_base_final, (new_width, new_height))

        stitched_image = warped_base_image

        for i in range(0, middle_index):
            current_image = processed_images_for_cv2[i]
            H = H_list[i]
            
            T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
            Homo_final = T @ H

            warped_image = cv2.warpPerspective(
                current_image,
                Homo_final,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,
            )
            h, w = current_image.shape[:2]  

            original_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            corner_trans = cv2.perspectiveTransform(original_corners, Homo_final)

            mask = warped_image != 0
            contour = corner_trans.reshape(1, 4, 2).astype(np.int32) 
            print(contour)
            boundary_mask = np.zeros((new_height, new_width), dtype=np.uint8)
            cv2.drawContours(boundary_mask, contour, -1, color=1, thickness=2)
            boundary_pixels = np.where(boundary_mask == 1) 
            mask[boundary_pixels] = False

            stitched_image[mask] = warped_image[mask]
            
        for i in range(middle_index + 1, len(H_list)):
            current_image = processed_images_for_cv2[i]
            H = H_list[i]
            
            T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
            Homo_final = T @ H

            warped_image = cv2.warpPerspective(
                current_image,
                Homo_final,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,
            )

            h, w = current_image.shape[:2]  
            original_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            corner_trans = cv2.perspectiveTransform(original_corners, Homo_final)

            mask = warped_image != 0
            contour = corner_trans.reshape(1, 4, 2).astype(np.int32)
            print(contour)
            boundary_mask = np.zeros((new_height, new_width), dtype=np.uint8)
            cv2.drawContours(boundary_mask, contour, -1, color=1, thickness=2)
            boundary_pixels = np.where(boundary_mask == 1)
            mask[boundary_pixels] = False

            stitched_image[mask] = warped_image[mask]

        cv2.imwrite(output, stitched_image)
        print(f"Stitched image saved to {output}")
        
visualize_tracks_on_images(
    images=images_vis,
    tracks=tracks,
    track_vis_mask=track_mask,
    out_dir="vggt_track_vis",
    image_format="CHW",
    normalize_mode="[0,1]",
    cmap_name="hsv",
    frames_per_row=2,
)

