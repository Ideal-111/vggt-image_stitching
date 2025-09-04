import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.activation_map import visualize_activation_map
from vggt.utils.attention_map import visualize_attention_map
from vggt.utils.keypoint import transform_keypoints, detectAndDescribe
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

num_registers = 4
model = VGGT()
# print(model)
state_dict = torch.load("/data/lx/ckpt/model.pt", map_location=device)
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

image_names = ['examples/hill/img1.jpg']
original_image_path = 'examples/hill/img1.jpg'

images = load_and_preprocess_images(image_names).to(device)
images_vis = images

# # SIFT关键点检测
# img1 = cv2.imread(image_names[0])
# kps1, features1 = detectAndDescribe(img1)

# inference
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None] 
        print(images.shape)
        aggregated_tokens_list, ps_idx = model.aggregator(images)

        attn_map = model.aggregator.global_blocks[-1].attn.last_attn
        print(attn_map.shape)
        cls_attention_all_tokens = attn_map.mean(dim=1)
        print(cls_attention_all_tokens.shape)
        attn_map_1d_for_patches = cls_attention_all_tokens[0, 0, 1 + num_registers:]
        print(attn_map_1d_for_patches.shape)
        print(f"提取后用于可视化的注意力图大小: {attn_map_1d_for_patches.size()}")

        # predict cameras
        # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # extrinsic and intrinsic metrices, following OpenCV convention(camera from world)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # predict depth map
        # depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # predict point map
        # point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

        # predict tracks
        # choose your own points to track, with shape (N, 2) for one scene

        query_points = torch.FloatTensor([[300, 100], [220, 300], [100, 50], [180, 350]]).to(device)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
        tracks = track_list[-1].squeeze(0)
        
visualize_activation_map(
    aggregated_tokens=aggregated_tokens_list,
    image_path=original_image_path,
    image_tensor=images,
    patch_size=model.aggregator.patch_size,
    ps_idx=ps_idx,
    alpha=0.9,
    cmap='viridis'
)

visualize_attention_map(
    image_tensor=images_vis,
    attn_map_1d_for_patches=attn_map_1d_for_patches,
    patch_size=model.aggregator.patch_size,
    alpha=0.9,    
    cmap_name='viridis',
    output_path='attention_visualization.png' 
    )

# print("相机外参(extrinsic):\n", extrinsic)
# print("相机内参(intrinsic):\n", intrinsic)
# print("深度图(depth_map):\n", depth_map)
# print("track_list:", track_list, "\nvis_score:", vis_score, "\nconf_score:", conf_score)
