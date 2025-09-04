import torch
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.visual_track import visualize_tracks_on_images
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
state_dict = torch.load("/data/lx/ckpt/model.pt", map_location=device)
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

image_names = ["examples/mix/20200708105115367_kjg.jpg", "examples/mix/20200708105115367_hw.jpg"]
images = load_and_preprocess_images(image_names).to(device)
images_vis = images
img1 = cv2.imread(image_names[0])

# inference
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None] 
        print(images.shape)
        aggregated_tokens_list, ps_idx = model.aggregator(images)

        # predict cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # extrinsic and intrinsic metrices, following OpenCV convention(camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # extrinsic_4x4 = torch.zeros((extrinsic.shape[0], 4, 4), device=extrinsic.device)
        # extrinsic_4x4[:, :3, :] = extrinsic
        # extrinsic_4x4[:, 3, 3] = 1

        # predict depth map
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # predict point map
        point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

        # predict tracks
        # choose your own points to track, with shape (N, 2) for one scene
        query_points = torch.FloatTensor([[300, 100], [220, 300], [100, 50], [180, 350]]).to(device)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
        print(track_list[-1].shape)
        tracks = track_list[-1].squeeze(0)
        print(tracks.shape)
        # vis = vis_score[0, -1]
        # conf = vis_score[0, -1]
        # mask = (vis > 0.70) & (conf > 0.50)
        # filtered_tracks = tracks[mask]    
        # print(filtered_tracks.shape)

visualize_tracks_on_images(
    images=images_vis,
    tracks=tracks,
    out_dir="vggt_track_vis",
    image_format="CHW",
    normalize_mode="[0,1]",
    cmap_name="hsv",
    frames_per_row=2,
)


# print("相机外参(extrinsic):\n", extrinsic)
# print("相机内参(intrinsic):\n", intrinsic)
# print("深度图(depth_map):\n", depth_map)
# print("track_list:", track_list, "\nvis_score:", vis_score, "\nconf_score:", conf_score)
