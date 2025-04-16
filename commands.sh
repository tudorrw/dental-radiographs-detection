pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# activate conda env
conda activate dental-dl

# split the data
python -m data.utils.train_test_val_split
python -m data.utils.train_test_split_for_kfolds

# commands for faster rcnn
python -m models.faster_rcnn.scripts.visualize_ground_truth
python -m models.faster_rcnn.scripts.train
python -m models.faster_rcnn.scripts.test
python -m models.faster_rcnn.scripts.predict
python -m models.faster_rcnn.params
tensorboard --logdir=checkpoints/faster_rcnn

# commands for detr
python -m models.detr.scripts.visualize_ground_truth
python -m models.detr.scripts.train
tensorboard --logdir=checkpoints/detr

# commands for yolo 11
python -m models.yolov11.scripts.train
python -m models.yolov11.scripts.val
python -m models.yolov11.scripts.test
tensorboard --logdir=runs/detect/train

# commands for retinanet
python -m models.retinanet.scripts.train
python -m models.retinanet.params
tensorboard --logdir=checkpoints/retinanet

# to vizualize the results of the yolo, faster_rcnn models, run the following command:
python -m models.visualize_results

python -m models.meta_model
#utils
# run frontend
cd ui
npm run dev

# run backend
uvicorn api.app:app --reload

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
