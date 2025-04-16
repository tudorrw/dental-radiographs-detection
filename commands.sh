pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# activate conda env
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
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

# to move the folder from local to the server, to call from local
scp -P <pid> -r <folder_to_move> root@<remoteip>:~/dental-radiographs-detection 
scp -r root@<remoteip>:/workspace/dental-radiographs-detection/checkpoints/faster_rcnn/version_0 <folder_to_move>

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
pip install torch==2.6.0+cu124 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu124

ln -s /workspace/miniconda3 ~/miniconda3
ln -s /workspace/dental-radiographs-detection ~/dental-radiographs-detection
ln -s /workspace/.gitconfig ~/.gitconfig

# requirements:
python-multipart
fastapi
ultralytics=8.3.96
numpy==2.1.2
opencv
albumentations==2.0.5
fastapi=0.115.12
seaborn=0.13.2
scikit-learn=1.6.1
tensorboard=2.19.0
tqdm=4.67.1
uvicorn=0.32.1
pillow==11.0.0
pycocotools==2.0.8
pandas=2.2.3
lightning=2.5.0
huggingface_hub=0.29.2