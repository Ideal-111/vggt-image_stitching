# VGGT-Image_stitching
Repository for running the VGGT model to do image stitching task
![example](https://github.com/Ideal-111/vggt-image_stitching/tree/main/stitched_processed_results/hill.jpg)

## TODO
 - Add procession of the seaming
 - Add end-to-end image stitching method
 - Adapt vggt to mutimodal image stitching task
## Installation

### Option1: pip installation
Same dependencies as VGGT([VGGT](https://github.com/facebookresearch/vggt)) installation

### Start
First, use [ALIKED](https://github.com/Shiaoming/ALIKED) to get the key points and save as .pt files, and then you can use the examples as follows:
```bash
python stitching/image_stitching.py
```
please check your file path in the code.

## License
The code is taken from the official [VGGT repository](https://github.com/facebookresearch/vggt) which is distributed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
See [LICENSE](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) for more information.


### References:
- vggt: https://github.com/facebookresearch/vggt
- ALIKED: https://github.com/Shiaoming/ALIKED
