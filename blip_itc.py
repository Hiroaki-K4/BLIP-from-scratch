import torch
import torch.nn as nn
import torch.nn.functional as F

from image_encoder import BLIPImageEncoder
from text_encoder import BLIPTextEncoder


class BLIP_ITC_MODEL(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.visual_encoder = BLIPImageEncoder(pretrained=True)
        self.text_encoder = BLIPTextEncoder()

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, image, input_ids, attention_mask):
        image_prj = self.visual_encoder(image)
        image_feat = F.normalize(image_prj, dim=-1)

        text_prj = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = F.normalize(text_prj, dim=-1)

        t = self.temp.exp()
        sim_i2t = image_feat @ text_feat.t() * t
        sim_t2i = text_feat @ image_feat.t() * t

        return sim_i2t, sim_t2i
