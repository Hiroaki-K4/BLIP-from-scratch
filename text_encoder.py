import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class BLIPTextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768, embed_dim=256):
        super().__init__()

        # Use standard BERT config for ITC task (bidirectional attention)
        self.config = BertConfig.from_pretrained(model_name)
        # For ITC task, we don't need cross-attention or decoder mode
        # self.config.add_cross_attention = True
        # self.config.is_decoder = True

        self.bert = BertModel.from_pretrained(model_name, config=self.config)
        self.text_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_ids, attention_mask, visual_embeds=None, mode="itc"):
        """
        mode 'itc': Feature extraction using text only(bidirectional), without image features
        mode 'itm': Inject image features and extract fusion features from text and images
        """
        if mode == "itc":
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            return self.text_proj(last_hidden_state)
        elif mode == "itm":
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=visual_embeds,
                return_dict=True,
            )
            return outputs.last_hidden_state


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BLIPTextEncoder().to(device)

    text = "A photo of a cat"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    dummy_visual_embeds = torch.randn(1, 50, 768).to(device)  # (batch, seq_len, dim)

    # 1. ITC mode
    text_feat = model(inputs.input_ids, inputs.attention_mask, mode="itc")
    print(f"ITC Feature Shape: {text_feat.shape}")  # [1, 256]

    # 2. ITM mode
    multimodal_feat = model(
        inputs.input_ids,
        inputs.attention_mask,
        visual_embeds=dummy_visual_embeds,
        mode="itm",
    )
    print(f"ITM Feature Shape: {multimodal_feat.shape}")  # [1, seq_len, 768]
