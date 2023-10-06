wget 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth' -O 'Wav2Lip-master/face_detection/detection/sfd/s3fd.pth'
pip install gdown
gdown https://drive.google.com/uc?id=1fQtBSYEyuai9MjBOF8j7zZ4oQ9W2N64q --output 'Wav2Lip-master/checkpoints/'
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P 'GFPGAN-master/experiments/pretrained_models'