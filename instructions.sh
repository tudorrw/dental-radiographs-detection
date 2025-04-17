#here are the instructions to set up the remote ip for the server in case your gpu sucks like mine
# or follow the instructions from this link:
https://www.youtube.com/watch?v=vEVDoW-uMHI&ab_channel=PromptEngineer

# 1. got the RunPod site
https://www.runpod.io/console/deploy

2. create a new pod
3. select the gpu you want (preferably RTX 4090 with Better Pytorch 2.6.0 CUDA12.4 https://www.runpod.io/console/explore/mm3gw8nlro)
4. on your local machine, create a new ssh key
    1. Create a key pair in a terminal window as follows:
    ssh-keygen -t ed25519

    2. Get your public key (you can use the following command if you used the defaults)
    cat ~/.ssh/id_ed25519.pub
    type %USERPROFILE%\.ssh\id_ed25519.pub

    3. Copy your SSH key to the server. For RunPod, you can find the menu in your settings in the top right corner.
# to move the folder from local to the server, to call from local
scp -P <pid> -r <folder_to_move> root@<remoteip>:~/dental-radiographs-detection 
scp -r root@<remoteip>:/workspace/dental-radiographs-detection/checkpoints/faster_rcnn/version_0 <folder_to_move>

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
pip install torch==2.6.0+cu124 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu124

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate dental-dl

ln -s /workspace/miniconda3 ~/miniconda3
ln -s /workspace/dental-radiographs-detection ~/dental-radiographs-detection
ln -s /workspace/.gitconfig ~/.gitconfig