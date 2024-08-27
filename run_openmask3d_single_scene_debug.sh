#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# RUN OPENMASK3D FOR A SINGLE SCENE

# Compute mask features for each mask and save them
echo "[INFO] Computing mask features..."

cd openmask3d

python compute_features_single_scene.py \
data.masks.masks_path='/home/tidy/PycharmProjects/SegmentAnything3D/save/mask_array/scene0671_00.pt' \
data.camera.poses_path='/home/tidy/PycharmProjects/SegmentAnything3D/scannetv2_val_images/scene0671_00/pose' \
data.camera.intrinsic_path='/home/tidy/PycharmProjects/SegmentAnything3D/scannetv2_val_images/scene0671_00/intrinsics/intrinsic_color.txt' \
data.camera.intrinsic_resolution="[968,1296]" \
data.depths.depths_path='/home/tidy/PycharmProjects/SegmentAnything3D/scannetv2_val_images/scene0671_00/depth' \
data.depths.depth_scale=1000 \
data.depths.depths_ext=".png" \
data.images.images_path='/home/tidy/PycharmProjects/SegmentAnything3D/scannetv2_val_images/scene0671_00/color' \
data.images.images_ext=".jpg" \
data.point_cloud_path="/home/tidy/PycharmProjects/SegmentAnything3D/pointcept_process/val/scene0671_00.ply" \
output.output_directory='/home/tidy/PycharmProjects/SegmentAnything3D/openmask3d/output/ours' \
hydra.run.dir="/home/tidy/PycharmProjects/SegmentAnything3D/openmask3d/output/ours/hydra_outputs/mask_features_computation" \
external.sam_checkpoint="/home/tidy/PycharmProjects/SegmentAnything3D/sam_vit_h_4b8939.pth" \
# echo "[INFO] Feature computation done!"
