import torch
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
state_dict = torch.load("/data/lx/ckpt/model.pt", map_location=device)
model.load_state_dict(state_dict)

model.eval()
model = model.to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["examples/room/images/no_overlap_1.png", "examples/room/images/no_overlap_2.jpg", "examples/room/images/no_overlap_3.jpg", 
               "examples/room/images/no_overlap_4.jpg", "examples/room/images/no_overlap_5.jpg", "examples/room/images/no_overlap_6.jpg",
               "examples/room/images/no_overlap_7.jpg", "examples/room/images/no_overlap_8.jpg"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        # print(predictions)
        
depth = predictions["depth"]  # [B, S, H, W, 1]
depth = depth.squeeze(-1).cpu().numpy()  # -> [B, S, H, W]
pose_enc = predictions["pose_enc"].cpu().numpy()
world_points = predictions["world_points"].cpu().numpy()
images_cpu = predictions["images"].cpu().numpy()

for b in range(depth.shape[0]):
    for s in range(depth.shape[1]):
        plt.imshow(depth[b, s], cmap='plasma')
        plt.title(f"Depth Map - Batch {b}, Frame {s+1}")
        plt.colorbar()
        plt.show()
        
