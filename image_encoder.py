import timm
import torch
import torch.nn as nn


class BLIPImageEncoder(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        hidden_dim=768,
        embed_dim=256,
        pretrained=True,
    ):
        super().__init__()
        self.visual_encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.vision_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        image_embeds = self.visual_encoder.forward_features(x)
        image_prj = self.vision_proj(image_embeds[:, 0, :])
        return image_embeds, image_prj


if __name__ == "__main__":
    model = BLIPImageEncoder(pretrained=True)
    model.eval()
    dummy_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        image_embeds, image_prj = model(dummy_img)
    print(
        f"Image embedding shape: {image_embeds.shape}, Image projection shape: {image_prj.shape}"
    )  # [1, 197, 768], [1, 256]
