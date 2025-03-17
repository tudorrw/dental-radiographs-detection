# source teeth-yolov11-venv/Scripts/activate
# cd process
python -m data.utils.train_test_val_split voc
python -m data.utils.train_test_val_split coco
python -m models.faster_rcnn.scripts.train
python -m models.ssd.scripts.train