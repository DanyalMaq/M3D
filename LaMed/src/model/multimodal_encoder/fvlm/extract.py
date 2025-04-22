import torch
 
# Load the checkpoint
ckpt = torch.load("model.pth", map_location="cpu")
 
# Grab the actual model weights (in this case it's under 'model')
state_dict = ckpt["model"]
 
# Filter out just the visual encoder weights
vit_encoder_weights = {
    k.replace("visual_encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("visual_encoder.")
}
 
# Save them for reuse
torch.save(vit_encoder_weights, "vit_encoder_only.pth")
print("Saved encoder weights with", len(vit_encoder_weights), "parameters.")