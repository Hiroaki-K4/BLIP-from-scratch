import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from blip import BLIP


def generate_caption(
    model,
    image_tensor,
    tokenizer,
    device,
    max_length=30,
    use_sampling=False,
    temperature=1.0,
    repetition_penalty=1.5,
):
    """Generate caption using LM mode - autoregressive text generation

    Args:
        use_sampling: If False, uses greedy decoding (deterministic, like BLIP)
                      If True, uses probabilistic sampling (non-deterministic)
        temperature: Only used when use_sampling=True
        repetition_penalty: Penalize repeated tokens (>1.0 reduces repetition)
    """
    model.eval()

    # Start with [CLS] token
    generated_ids = [tokenizer.cls_token_id]

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare current sequence
            input_ids = torch.tensor([generated_ids]).to(device)
            attention_mask = torch.ones_like(input_ids)

            # Get next token prediction
            lm_logits = model(image_tensor, input_ids, attention_mask, mode="lm")

            # Get logits for the last position (next token)
            next_token_logits = lm_logits[0, -1, :].clone()

            # Apply repetition penalty to already generated tokens
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    # If token has positive logit, divide by penalty
                    # If token has negative logit, multiply by penalty
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty

            # Select next token
            if use_sampling:
                # Probabilistic sampling (non-deterministic)
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding (deterministic, like BLIP official)
                next_token_id = torch.argmax(next_token_logits).item()

            # Stop if we hit [SEP] or [PAD]
            if next_token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                break

            generated_ids.append(next_token_id)

    # Convert to text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def load_model(model_path, device):
    model = BLIP()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def inference(model_path, image_path, text_candidates, mode="itc"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Only tokenize text_candidates if not in LM mode
    if mode != "lm":
        inputs = tokenizer(text_candidates, padding=True, return_tensors="pt").to(
            device
        )

    with torch.no_grad():
        if mode == "itc":
            # ITC mode: Image-Text Contrastive
            sim_i2t, _ = model(
                image_tensor, inputs.input_ids, inputs.attention_mask, mode="itc"
            )
            logits = sim_i2t[0]  # (num_texts,)
            probs = F.softmax(logits, dim=-1)

        elif mode == "itm":
            # ITM mode: Image-Text Matching
            # Process each text-image pair individually for proper discrimination
            # Batch processing causes identical scores due to cross-attention interference
            itm_scores = []
            for i, text in enumerate(text_candidates):
                text_input_ids = inputs.input_ids[i : i + 1]  # Single text
                text_attention_mask = inputs.attention_mask[i : i + 1]  # Single text

                itm_output = model(
                    image_tensor, text_input_ids, text_attention_mask, mode="itm"
                )

                # Use raw matching logit to preserve subtle but meaningful differences
                # ITM outputs [no_match_logit, match_logit] - we want the match score
                matching_logit = itm_output[
                    0, 1
                ].item()  # Raw matching logit (can be negative)
                itm_scores.append(matching_logit)

            probs = torch.tensor(itm_scores)  # Raw logits preserve discrimination

        elif mode == "lm":
            # LM mode: Language Model - Generate caption from image
            # Use greedy decoding (deterministic, like BLIP official)
            generated_caption = generate_caption(
                model,
                image_tensor,
                tokenizer,
                device,
                max_length=30,
                use_sampling=False,
            )
            return generated_caption

    print(f"\n--- {mode.upper()} Inference Result ({image_path}) ---")
    for i, text in enumerate(text_candidates):
        print(f"Score: {probs[i].item():.4f} | Text: {text}")

    best_idx = probs.argmax().item()
    print(f"\nBest Match: {text_candidates[best_idx]}")
    return probs


if __name__ == "__main__":
    model_path = "best_blip_model.pth"
    image_path = "original/caption.jpg"
    text_candidates = [
        "A sunset over the ocean",
        "A man riding a motor bike on a dirt road on the countryside",
        "A person playing the guitar",
        "A futuristic city with flying cars",
    ]

    print("====== BLIP Inference Test ======")

    # ITC mode inference
    print("\n🔍 ITC (Image-Text Contrastive) Mode:")
    itc_probs = inference(model_path, image_path, text_candidates, mode="itc")

    # ITM mode inference
    print("\n🤖 ITM (Image-Text Matching) Mode:")
    itm_probs = inference(model_path, image_path, text_candidates, mode="itm")

    # LM mode inference
    print("\n✨ LM (Language Model) Mode - Caption Generation:")
    generated_caption = inference(model_path, image_path, [], mode="lm")

    # Comparison results
    print("\n📊 Comparison Results:")
    print("Text Candidate | ITC Score | ITM Score")
    print("-" * 45)
    for i, text in enumerate(text_candidates):
        print(f"{text:<25} | {itc_probs[i].item():.4f}   | {itm_probs[i].item():.4f}")

    print(f"\n🏆 Results Summary:")
    print(f"ITC Best Match: {text_candidates[itc_probs.argmax()]}")
    print(f"ITM Best Match: {text_candidates[itm_probs.argmax()]}")
    print(f"LM Generated Caption: {generated_caption}")
