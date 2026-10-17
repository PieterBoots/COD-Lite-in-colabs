import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download

# --- 1. Clone official GenCodec Repository ---
print('--- Cloning official GenCodec Repository ---')
if os.path.exists('GenCodec'):
    shutil.rmtree('GenCodec')
if os.path.exists('CoD_Lite'):
    shutil.rmtree('CoD_Lite')

!git clone https://github.com/microsoft/GenCodec.git
%cd GenCodec/CoD_Lite

# --- 2. Install Requirements ---
print('\n--- Installing Official Requirements ---')
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        reqs = f.read()
    reqs = reqs.replace('torchvision==0.20.0', 'torchvision')
    reqs = reqs.replace('torch==2.5.0', 'torch')
    with open('requirements.txt', 'w') as f:
        f.write(reqs)

!pip install -q -r requirements.txt
!pip install -q huggingface_hub

# --- 3. Download Checkpoint (0.5000 BPP) ---
print('\n--- Downloading CoD-Lite Checkpoint ---')
os.makedirs('pretrained/CoD_Lite', exist_ok=True)
hf_hub_download(repo_id='zhaoyangjia/CoD_Lite', filename='CoD_Lite_bpp_0_5000.pt', local_dir='./pretrained/CoD_Lite')
hf_hub_download(repo_id='zhaoyangjia/CoD_Lite', filename='CoD_Lite_bpp_0_5000.yaml', local_dir='./pretrained/CoD_Lite')

# --- 4. Prepare Directories and Input Image ---
print('\n--- Preparing Test Directories ---')
shutil.rmtree('colab_inputs', ignore_errors=True)
shutil.rmtree('colab_outputs', ignore_errors=True)
os.makedirs('colab_inputs', exist_ok=True)
os.makedirs('colab_outputs', exist_ok=True)

try:
    user_img_path = '/content/bike.bmp'
    if os.path.exists(user_img_path):
        img = Image.open(user_img_path).convert('RGB')
        img.save('colab_inputs/bike.jpg', 'JPEG')
        print('Successfully prepared colab_inputs/bike.jpg')
    else:
        raise FileNotFoundError('bike.bmp not found at /content/bike.bmp')
except Exception as e:
    print(f'Using fallback image: {e}')
    !wget -q -O colab_inputs/bike.jpg https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png

# --- 5. Run Inference ---
print('\n--- Running CoD-Lite Evaluation ---')
!python -m finetuned_one_step_codec.inference evaluate \
    --ckpt ./pretrained/CoD_Lite/CoD_Lite_bpp_0_5000.pt \
    --config ./pretrained/CoD_Lite/CoD_Lite_bpp_0_5000.yaml \
    --input ./colab_inputs \
    --output ./colab_outputs

# --- 6. Plot Results ---
try:
    orig_path = 'colab_inputs/bike.jpg'
    recon_files = [f for f in os.listdir('colab_outputs') if f.endswith(('.png', '.jpg', '.jpeg'))]
    if recon_files:
        recon_path = os.path.join('colab_outputs', recon_files[0])
        orig_img = Image.open(orig_path)
        recon_img = Image.open(recon_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(orig_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(recon_img)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        plt.show()
except Exception as e:
    print(f'Visualization error: {e}')
