# source teeth-yolov11-venv/Scripts/activate
# cd process
python -m process.train_test_val_split
python -m models.faster_rcnn.scripts.train
python -m models.ssd.ssd