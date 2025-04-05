# source teeth-yolov11-venv/Scripts/activate
# cd process
python -m data.utils.train_test_val_split
python -m data.utils.train_test_val_split

python -m models.faster_rcnn.scripts.train
python -m models.faster_rcnn.scripts.test
python -m models.faster_rcnn.scripts.predict

python -m models.yolov11.scripts.train
python -m models.detr.scripts.train
python -m models.ssd.scripts.train

python -m models.detr.scripts.visualize_ground_truth
python -m models.faster_rcnn.scripts.visualize_ground_truth
#utils
conda activate dental-dl
tensorboard --logdir=checkpoints/faster_rcnn

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
tensorboard --logdir=runs/detect/train