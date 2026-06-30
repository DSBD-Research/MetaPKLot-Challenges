# MetaPKLot

## Requirements

```
conda create --name pklot python=3.10 -y
conda activate pklot

conda install pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install --force-reinstall mkl=2023.1 mkl-service numpy -c defaults -y
pip install openmim
pip install chardet
mim install mmengine
mim install mmcv==2.0.1
mim install mmdet
mim install mmpretrain
pip install numpy==1.26.4
pip install faster_coco_eval scikit-learn

conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.6.1  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install pycocotools
```

## Setup

1. Execute `tools/setup.py`. For example: `python3 setup.py ../../MetaPKLot-Dataset/ ../../PLds_converted/ ../../metapklot`
2. In the example above, change `../../metapklot/cjsons/isshk_cars.json` and `../../metapklot/cjsons/isshk_spots.json` with the JSON files of `MetaPKLot-Dataset/annotations/cropped/cars/PLds/isshk_cars.json` and `MetaPKLot-Dataset/annotations/cropped/spots/PLds/isshk_spots.json`.
3. Adjust the configs in `configs/` according to the paths in your system.

## Example of running

Here is an example for running the entire pipeline for the experiment "wo_PLds_001". Change the gpu index using the `CUDA_VISIBLE_DEVICES`.

```
CUDA_VISIBLE_DEVICES=0 nohup python train.py configs/wo_PLds.py --work-dir /disk1/lmk22/parking_spot_recognition/work_dirs/wo_plds_001/ &
```

```
CUDA_VISIBLE_DEVICES=0 nohup python generate_preds.py --output_dir /disk1/lmk22/parking_spot_recognition/instance_seg/ --annotations_dir /disk1/lmk22/parking_spot_recognition/metapklot/cjsons/ --root /disk1/lmk22/parking_spot_recognition/metapklot/ --model_dir /disk1/lmk22/parking_spot_recognition/work_dirs/ --exp wo_plds_001 &
```

```
nohup python generate_graph.py --output_dir /disk1/lmk22/parking_spot_recognition/instance_seg/ --annotations_dir /disk1/lmk22/parking_spot_recognition/metapklot/cjsons/ --root /disk1/lmk22/parking_spot_recognition/metapklot/ --model_dir /disk1/lmk22/parking_spot_recognition/parking_spot_extraction/work_dirs/ --exp wo_plds_001 &
```

```
CUDA_VISIBLE_DEVICES=0 nohup python train_graphspot.py --output_dir /disk1/lmk22/parking_spot_recognition/instance_seg/ --root /disk1/lmk22/parking_spot_recognition/metapklot/cjsons/ --exp wo_plds_001 &
```

```
python test_graphspot.py --output_dir /disk1/lmk22/parking_spot_recognition/instance_seg/ --annotations_dir /disk1/lmk22/parking_spot_recognition/metapklot/cjsons/ --exp wo_plds_001
```
