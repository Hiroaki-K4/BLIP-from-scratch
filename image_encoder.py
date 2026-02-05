import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class BLIPImageEncoder(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
    ):
        super().__init__()
        self.visual_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
        )

    def forward(self, x):
        return self.visual_encoder.forward_features(x)


if __name__ == "__main__":
    model = BLIPImageEncoder()
    dummy_img = torch.randn(1, 3, 224, 224)
    features = model(dummy_img)
    print(
        f"Feature shape: {features.shape}"
    )  # [1, 197, 768] (196 patches + 1 cls token)
