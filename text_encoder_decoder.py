import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class BLIPModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768, embed_dim=256):
        super().__init__()

        self.base_bert = BertModel.from_pretrained(model_name)
        self.config = self.base_bert.config

        common_config = BertConfig.from_pretrained(model_name)
        common_config.add_cross_attention = True
        common_config.is_decoder = True
        self.itm_bert = BertModel.from_pretrained(model_name, config=common_config)

        self.text_decoder = BertModel.from_pretrained(model_name, config=common_config)

        self._share_weights()

        self.text_proj = nn.Linear(hidden_dim, embed_dim)
        self.itm_head = nn.Linear(hidden_dim, 2)
        self.lm_head = nn.Linear(hidden_dim, self.config.vocab_size)

    def _share_weights(self):
        """Share weights except cross attention"""
        self.itm_bert.embeddings = self.base_bert.embeddings
        self.text_decoder.embeddings = self.base_bert.embeddings

        for i in range(len(self.base_bert.encoder.layer)):
            base_layer = self.base_bert.encoder.layer[i]
            itm_layer = self.itm_bert.encoder.layer[i]
            dec_layer = self.text_decoder.encoder.layer[i]

            # Share layers except cross attention between ITC and ITM
            itm_layer.attention.self = base_layer.attention.self
            itm_layer.attention.output = base_layer.attention.output
            itm_layer.intermediate = base_layer.intermediate
            itm_layer.output = base_layer.output

            # Share cross attention between ITM and decoder
            dec_layer.intermediate = base_layer.intermediate
            dec_layer.output = base_layer.output
            dec_layer.crossattention = itm_layer.crossattention

    def forward(self, input_ids, attention_mask, visual_embeds=None, mode="itc"):
        """
        mode 'itc': Feature extraction using text only(bidirectional), without image features
        mode 'itm': Inject image features and extract fusion features from text and images
        """
        if mode == "itc":
            outputs = self.base_bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            return self.text_proj(last_hidden_state)
        elif mode == "itm":
            # outputs = self.itm_bert(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     encoder_hidden_states=visual_embeds,
            # )
            # extended_attention_mask = self.base_bert.get_extended_attention_mask(
            #     attention_mask, input_ids.size(), device=input_ids.device
            # )
            # outputs = self.itm_bert(
            #     input_ids=input_ids,
            #     attention_mask=extended_attention_mask,
            #     encoder_hidden_states=visual_embeds,
            # )
            embedding_output = self.itm_bert.embeddings(input_ids=input_ids)
            extended_attention_mask = self.base_bert.get_extended_attention_mask(
                attention_mask, input_ids.size()
            )
            encoder_outputs = self.itm_bert.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=visual_embeds,
            )
            return self.itm_head(encoder_outputs.last_hidden_state[:, 0, :])
            # return self.itm_head(outputs.last_hidden_state[:, 0, :])
        elif mode == "lm":
            outputs = self.text_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=visual_embeds,
            )
            return self.lm_head(outputs.last_hidden_state)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BLIPModel().to(device)

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
    print(f"ITM Feature Shape: {multimodal_feat.shape}")  # [1, 2]

    # 3. LM mode
    lm_logits = model(
        inputs.input_ids,
        inputs.attention_mask,
        visual_embeds=dummy_visual_embeds,
        mode="lm",
    )
    print(f"LM Logits Shape: {lm_logits.shape}")  # [1, seq_len, vocab_size]
