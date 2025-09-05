import os
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.utils.keypoint import transform_keypoints
from vggt.utils.stitching import stitch_images
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

        stitched_image = stitch_images(full_path_image_names, tracked_pts, processed_images_for_cv2)

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