"""
Visualize BLIP inference results with images and scores
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from infer import generate_caption, load_model


def visualize_inference(
    model_path, image_path, text_candidates, output_path="inference_result.png"
):
    """Generate a visual comparison of BLIP inference modes"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model = load_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Preprocess image
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
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Get ITC scores
    inputs = tokenizer(text_candidates, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        sim_i2t, _ = model(
            image_tensor, inputs.input_ids, inputs.attention_mask, mode="itc"
        )
        itc_probs = F.softmax(sim_i2t[0], dim=-1).cpu().numpy()

    # Get ITM scores
    itm_scores = []
    with torch.no_grad():
        for i in range(len(text_candidates)):
            text_input_ids = inputs.input_ids[i : i + 1]
            text_attention_mask = inputs.attention_mask[i : i + 1]
            itm_output = model(
                image_tensor, text_input_ids, text_attention_mask, mode="itm"
            )
            itm_scores.append(itm_output[0, 1].item())
    itm_scores = torch.tensor(itm_scores).numpy()

    # Generate caption
    generated_caption = generate_caption(
        model,
        image_tensor,
        tokenizer,
        device,
        max_length=30,
        use_sampling=False,
        repetition_penalty=1.5,
    )

    # Create visualization
    fig = plt.figure(figsize=(16, 10))

    # Image display at the top
    ax_img = plt.subplot(3, 1, 1)
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(
        f'Generated Caption: "{generated_caption}"',
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # ITC scores
    ax_itc = plt.subplot(3, 2, 3)
    colors_itc = [
        "#4CAF50" if i == itc_probs.argmax() else "#90CAF9"
        for i in range(len(text_candidates))
    ]
    bars_itc = ax_itc.barh(range(len(text_candidates)), itc_probs, color=colors_itc)
    ax_itc.set_yticks(range(len(text_candidates)))
    ax_itc.set_yticklabels(
        [f"{t[:35]}..." if len(t) > 35 else t for t in text_candidates], fontsize=10
    )
    ax_itc.set_xlabel("Probability", fontsize=11)
    ax_itc.set_title(
        "ITC (Image-Text Contrastive) Scores", fontsize=12, fontweight="bold"
    )
    ax_itc.set_xlim(0, max(itc_probs) * 1.2)
    for i, (bar, score) in enumerate(zip(bars_itc, itc_probs)):
        ax_itc.text(
            score,
            bar.get_y() + bar.get_height() / 2,
            f" {score:.3f}",
            va="center",
            fontsize=9,
        )

    # ITM scores
    ax_itm = plt.subplot(3, 2, 4)
    colors_itm = [
        "#4CAF50" if i == itm_scores.argmax() else "#FFB74D"
        for i in range(len(text_candidates))
    ]
    bars_itm = ax_itm.barh(range(len(text_candidates)), itm_scores, color=colors_itm)
    ax_itm.set_yticks(range(len(text_candidates)))
    ax_itm.set_yticklabels(
        [f"{t[:35]}..." if len(t) > 35 else t for t in text_candidates], fontsize=10
    )
    ax_itm.set_xlabel("Matching Score (logit)", fontsize=11)
    ax_itm.set_title("ITM (Image-Text Matching) Scores", fontsize=12, fontweight="bold")
    for i, (bar, score) in enumerate(zip(bars_itm, itm_scores)):
        ax_itm.text(
            score,
            bar.get_y() + bar.get_height() / 2,
            f" {score:.3f}",
            va="center",
            fontsize=9,
        )

    # Comparison table
    ax_table = plt.subplot(3, 1, 3)
    ax_table.axis("off")

    # Best matches summary
    itc_best = text_candidates[itc_probs.argmax()]
    itm_best = text_candidates[itm_scores.argmax()]

    summary_text = f"""
    Results Summary
    
    ITC Best Match: {itc_best} (Probability: {itc_probs.max():.4f})
    
    ITM Best Match: {itm_best} (Matching Score: {itm_scores.max():.4f})
    
    Generated Caption: {generated_caption}
    
    Note: ITC uses contrastive learning (similarity), ITM uses cross-attention (matching), 
    and LM generates captions autoregressively.
    """

    ax_table.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    plt.close()

    return generated_caption, itc_probs, itm_scores


if __name__ == "__main__":
    model_path = "best_blip_model.pth"
    image_path = "original/caption.jpg"
    text_candidates = [
        "A sunset over the ocean",
        "A man riding a motor bike on a dirt road on the countryside",
        "A person playing the guitar",
        "A futuristic city with flying cars",
    ]

    print("Generating BLIP inference visualization...")
    visualize_inference(model_path, image_path, text_candidates, "inference_result.png")
    print("\nDone! Check 'inference_result.png' for the visual results.")
