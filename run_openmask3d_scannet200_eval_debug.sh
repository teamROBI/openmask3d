#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

WS_PATH="/home/robi/PycharmProjects/SegmentAnything3D"

# OPENMASK3D SCANNET200 EVALUATION SCRIPT
# This script performs the following in order to evaluate OpenMask3D predictions on the ScanNet200 validation set
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them
# 3. Evaluate for closed-set 3D semantic instance segmentation

# --------
# NOTE: SET THESE PARAMETERS!

# input directories of scannetv2 data
SCANS_PATH="${WS_PATH}/data/scannetv2/input/pointcept_process/val" # 3D data
SCANNET_PROCESSED_DIR="${WS_PATH}/data/scannetv2/processed_data/scannetv2_200_openmask3d" # 3D data with labeling
SCANS_2D_PATH="${WS_PATH}/data/scannetv2/input/scannetv2_images/val" # 2D data

SAM_CKPT_PATH="${WS_PATH}/weights/sam_vit_h_4b8939.pth"

# output directories to save masks and mask features
EXPERIMENT_NAME="scannetv2_200"
MODEL="cropformer_pc_depth"
MERGING_PARAMETERS="small_overlap_0.5" #ov_large_overlap_0.3, large_overlap_0.3, small_overlap_0.5, save_bivi, save

OUTPUT_DIRECTORY="$(pwd)/output/${EXPERIMENT_NAME}/${MODEL}/${MERGING_PARAMETERS}"
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/validation"
MASK_FEATURE_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"
EVALUATION_OUTPUT_DIR="${OUTPUT_FOLDER_DIRECTORY}/evaluation_result.txt"

# input directory where class-agnositc masks are saved
MASK_SAVE_DIR="${WS_PATH}/output/scannetv2/${MERGING_PARAMETERS}/mask_array/$MODEL"

# Paremeters below are AUTOMATICALLY set based on the parameters above:
SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
SCANNET_INSTANCE_GT_DIR="${SCANNET_PROCESSED_DIR%/}/instance_gt/validation"

# Set frequency to see how much image you want.
FREQUENCY=10

# Set to true if you wish to save the 2D crops of the masks from which the CLIP features are extracted. It can be helpful for debugging and for a qualitative evaluation of the quality of the masks.
SAVE_CROPS=true

# gpu optimization
OPTIMIZE_GPU_USAGE=false

cd openmask3d

# 2. Compute mask features
# echo "[INFO] Computing mask features..."
# python compute_features_scannet200.py \
# data.scans_path=${SCANS_PATH} \
# data.masks.masks_path=${MASK_SAVE_DIR} \
# data.scans_2d_path=${SCANS_2D_PATH}  \
# output.output_directory=${MASK_FEATURE_SAVE_DIR} \
# output.experiment_name=${EXPERIMENT_NAME} \
# output.save_crops=${SAVE_CROPS} \
# external.sam_checkpoint=${SAM_CKPT_PATH} \
# openmask3d.frequency=${FREQUENCY} \
# gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
# hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation"
# echo "[INFO] Feature computation done!"

# 3. Evaluate for closed-set 3D semantic instance segmentation
echo "[INFO] Evaluating..."
python evaluation/run_eval_close_vocab_inst_seg.py \
--gt_dir=${SCANNET_INSTANCE_GT_DIR} \
--mask_pred_dir=${MASK_SAVE_DIR} \
--mask_features_dir=${MASK_FEATURE_SAVE_DIR} \
--evaluation_output_dir=${EVALUATION_OUTPUT_DIR}
