from vit import ViT
import torch

vision_model_weights = torch.load("vit_encoder_only.pth", map_location='cpu')
vision = ViT(img_size=(112,256,352), patch_size=(16,16,32), in_channels=1, qkv_bias=True)
vision.load_state_dict(vision_model_weights)
print(vision)