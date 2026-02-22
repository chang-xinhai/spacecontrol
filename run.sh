# instructions for your setup: https://pytorch.org/get-started/locally/
# pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# pip install xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128
# pip install flash-attn --no-build-isolation
# pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
# pip install spconv-cu120

# mkdir -p /tmp/extensions
# git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
# pip install /tmp/extensions/nvdiffrast --no-build-isolation

# git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
# pip install /tmp/extensions/diffoctreerast --no-build-isolation

# git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
# pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# cp -r extensions/vox2seq /tmp/extensions/vox2seq
# pip install /tmp/extensions/vox2seq --no-build-isolation


export HF_ENDPOINT=https://hf-mirror.com
hf download --no-force-download microsoft/TRELLIS-image-large --local-dir ckpt/microsoft/TRELLIS-image-large
hf download --no-force-download microsoft/TRELLIS-text-xlarge --local-dir ckpt/microsoft/TRELLIS-text-xlarge

hf download --no-force-download openai/clip-vit-large-patch14 --local-dir ckpt/openai/clip-vit-large-patch14

# wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth -O ./ckpt/facebook/dinov2_vitl14_reg4_pretrain.pth