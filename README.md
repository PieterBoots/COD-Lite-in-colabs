```markdown
# CoD-Lite Image Compression & Reconstruction

This repository evaluates the official Microsoft **CoD-Lite** model at a 0.5000 BPP target.

## Features
- Automated GenCodec setup and framework requirements patching
- Downloads official 0.5000 BPP checkpoints from Hugging Face
- Pre-processes custom input formats into standard RGB JPEGs
- Generates reconstructed outputs using one-step diffusion

## Quick Start
```bash
git clone https://github.com/microsoft/GenCodec.git
cd GenCodec/CoD_Lite
pip install -r requirements.txt
python -m finetuned_one_step_codec.inference evaluate \
    --ckpt ./pretrained/CoD_Lite/CoD_Lite_bpp_0_5000.pt \
    --config ./pretrained/CoD_Lite/CoD_Lite_bpp_0_5000.yaml \
    --input ./colab_inputs \
    --output ./colab_outputs
```
```
