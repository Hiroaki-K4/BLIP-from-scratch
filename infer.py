import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from blip import BLIP_MODEL


def load_model(model_path, device):
    model = BLIP_MODEL()
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

    inputs = tokenizer(text_candidates, padding=True, return_tensors="pt").to(device)

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
        "A man petting a dog near garlic boxes",
        "A person riding a bicycle",
        "A futuristic city with flying cars",
    ]

    print("====== BLIP Inference Test ======")

    # ITC mode inference
    print("\n🔍 ITC (Image-Text Contrastive) Mode:")
    itc_probs = inference(model_path, image_path, text_candidates, mode="itc")

    # ITM mode inference
    print("\n🤖 ITM (Image-Text Matching) Mode:")
    itm_probs = inference(model_path, image_path, text_candidates, mode="itm")

    # Comparison results
    print("\n📊 Comparison Results:")
    print("Text Candidate | ITC Score | ITM Score")
    print("-" * 45)
    for i, text in enumerate(text_candidates):
        print(f"{text:<25} | {itc_probs[i].item():.4f}   | {itm_probs[i].item():.4f}")

    print(f"\nITC Best Match: {text_candidates[itc_probs.argmax()]}")
    print(f"ITM Best Match: {text_candidates[itm_probs.argmax()]}")
