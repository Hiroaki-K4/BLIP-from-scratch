import itertools

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from blip import BLIP
from dataloader import get_dataloader


def compute_losses(model, images, input_ids, attention_mask, device):
    """Compute all losses (ITC, ITM, LM) for given batch"""
    batch_size = images.size(0)

    # ITC loss
    sim_i2t, sim_t2i = model(images, input_ids, attention_mask, mode="itc")
    targets = torch.arange(batch_size).to(device)
    loss_i2t = F.cross_entropy(sim_i2t, targets)
    loss_t2i = F.cross_entropy(sim_t2i, targets)
    loss_itc = (loss_i2t + loss_t2i) / 2

    # ITM loss
    # Positive pair
    itm_output_pos = model(images, input_ids, attention_mask, mode="itm")
    # Negative pair
    input_ids_neg = torch.roll(input_ids, shifts=1, dims=0)
    attn_mask_neg = torch.roll(attention_mask, shifts=1, dims=0)
    itm_output_neg = model(images, input_ids_neg, attn_mask_neg, mode="itm")

    # Combine logits and labels
    itm_logits = torch.cat([itm_output_pos, itm_output_neg], dim=0)
    itm_labels = torch.cat(
        [
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
        ],
        dim=0,
    ).to(device)
    loss_itm = F.cross_entropy(itm_logits, itm_labels)

    # LM loss (Language Modeling)
    # Create labels for language modeling (shifted input_ids)
    lm_labels = input_ids.clone()
    lm_labels[:, 0] = -100  # Ignore [CLS] token
    lm_labels[attention_mask == 0] = -100  # Ignore padding tokens

    # Get LM predictions
    lm_logits = model(images, input_ids, attention_mask, mode="lm")
    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = lm_labels[:, 1:].contiguous()

    # Calculate language modeling loss
    loss_lm = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss_itc, loss_itm, loss_lm


def validate(model, dataloader, device, num_val_steps=50):
    """Calculate loss for validation data"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_iter = itertools.islice(dataloader, num_val_steps)
        for i, (images, input_ids, attention_mask) in enumerate(val_iter):
            images, input_ids, attention_mask = (
                images.to(device),
                input_ids.to(device),
                attention_mask.to(device),
            )

            # Compute all losses using the shared function
            loss_itc, loss_itm, loss_lm = compute_losses(
                model, images, input_ids, attention_mask, device
            )

            val_loss += (loss_itc + loss_itm + loss_lm).item()

    model.train()
    return val_loss / num_val_steps


def train(
    save_path="best_blip_model.pth",
    max_steps=500,
    batch_size=12,
    learning_rate=3e-4,  # ViT-B learning rate as per BLIP paper
    weight_decay=0.05,
    val_interval=200,
    patience=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BLIP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # BLIP paper settings: warm-up + exponential decay with rate 0.85
    warmup_steps = max_steps // 10  # 10% of training for warmup

    # More standard approach: Use ExponentialLR after warmup
    def lr_lambda(step):
        if step < warmup_steps:
            # Warm-up phase: linear increase
            return float(step) / float(max(1, warmup_steps))
        else:
            # After warmup, use exponential decay
            # Decay by 0.85 every 1000 steps (approximately every epoch for large datasets)
            decay_interval = 1000
            decay_count = (step - warmup_steps) // decay_interval
            return 0.85**decay_count

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="train"
    )
    val_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="validation"
    )
    train_iter = itertools.islice(train_loader, max_steps)

    # Variables for Early Stopping
    best_val_loss = float("inf")
    counter = 0

    model.train()
    total_loss = 0

    for i, (images, input_ids, attention_mask) in enumerate(train_iter):
        images, input_ids, attention_mask = (
            images.to(device),
            input_ids.to(device),
            attention_mask.to(device),
        )

        # Compute all losses using the shared function
        loss_itc, loss_itm, loss_lm = compute_losses(
            model, images, input_ids, attention_mask, device
        )

        # Total loss: combine all three losses
        loss = loss_itc + loss_itm + loss_lm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        total_loss += loss.item()

        if i > 0 and i % val_interval == 0:
            avg_train_loss = total_loss / (i + 1)
            current_val_loss = validate(model, val_loader, device)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nStep: {i} | Train Loss: {avg_train_loss:.4f} | Val Loss: {current_val_loss:.4f} | LR: {current_lr:.2e}"
            )
            print(
                f"  ITC Loss: {loss_itc.item():.4f} | ITM Loss: {loss_itm.item():.4f} | LM Loss: {loss_lm.item():.4f}"
            )

            # Early Stopping
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"--> Best model saved (Loss improved).")
            else:
                counter += 1
                print(
                    f"--> No improvement. EarlyStopping counter: {counter}/{patience}"
                )

            if counter >= patience:
                print("Early stopping triggered. Training finished.")
                break

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train(
        save_path="best_blip_model.pth",
        max_steps=10000,
        batch_size=16,
        weight_decay=0.05,
        val_interval=200,
        patience=3,
    )
