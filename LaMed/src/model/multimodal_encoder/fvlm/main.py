from vit import ViT
import torch

vision_model_weights = torch.load("vit_encoder_only.pth", map_location='cpu')
vision = ViT(img_size=(112,256,352), patch_size=(16,16,32), in_channels=1, qkv_bias=True, pos_embed="conv", classification=True)
# for ((k1, v1), (k2, v2)) in zip(vision.state_dict().items(), vision_model_weights.items()):
#     print(f"{k1}: {v1.shape}")
#     print(f"{k2}: {v2.shape}")
# vision.load_state_dict(vision_model_weights)

t = torch.randn(2, 1, 112, 256, 352)
vision.forward(t, t)