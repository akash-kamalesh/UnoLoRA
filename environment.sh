mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
# source ~/.zshrc~
source ~/.bashrc
sudo apt update -y
sudo apt install nvidia-cuda-toolkit -y
conda create -n unolora python=3.10 -y
conda activate unolora
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge ipykernel -y
python -m ipykernel install --user --name=unolora
conda install -c conda-forge jupyterlab -y
conda install -c conda-forge nodejs -y
pip install ipywidgets 
pip install transformers huggingface_hub datasets adapters wandb 
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install numpy==1.26.4 evaluate tabulate scikit-learn wandb matplotlib peft bitsandbytes sentence_transformers
pip install spacy
pip install SceneGraphParser
pip install numpy==1.26.4
python -m spacy download en_core_web_sm
pip install ftfy
pip install easydict 
pip install open_clip_torch
pip install opencv-python
pip install word2number
pip install setuptools==69.5.1
pip install git+https://github.com/openai/CLIP.git
sudo apt install htop zip unzip -y

scp -C -P 52310 -c aes128-ctr images.zip root@174.88.114.240:/root
