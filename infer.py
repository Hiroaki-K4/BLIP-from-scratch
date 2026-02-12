import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from blip_itc import BLIP_ITC_MODEL


def load_model(model_path, device):
    model = BLIP_ITC_MODEL()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def itc_inference(model_path, image_path, text_candidates):
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
        num_texts = len(text_candidates)
        expanded_images = image_tensor.repeat(num_texts, 1, 1, 1)

        sim_i2t, _ = model(expanded_images, inputs.input_ids, inputs.attention_mask)

        logits = sim_i2t[0]
        probs = F.softmax(logits, dim=-1)

    print(f"\n--- Inference Result ({image_path}) ---")
    for i, text in enumerate(text_candidates):
        print(f"Score: {probs[i].item():.4f} | Text: {text}")

    best_idx = probs.argmax().item()
    print(f"\nBest Match: {text_candidates[best_idx]}")


if __name__ == "__main__":
    model_path = "best_blip_itc_model.pth"
    image_path = "original/caption.jpg"
    text_candidates = [
        "A man is petting a dog in a barn",
        "A sunset over the ocean",
        "A person riding a bicycle",
        "A futuristic city with flying cars",
    ]
    itc_inference(model_path, image_path, text_candidates)
