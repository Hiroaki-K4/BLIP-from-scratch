import itertools

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from blip import BLIP
from dataloader import get_dataloader


def compute_losses(
    model, images, input_ids, attention_mask, device, use_hard_negatives=True
):
    """Compute all losses (ITC, ITM, LM) for given batch"""
    batch_size = images.size(0)

    # ITC loss
    sim_i2t, sim_t2i = model(images, input_ids, attention_mask, mode="itc")
    targets = torch.arange(batch_size).to(device)
    loss_i2t = F.cross_entropy(sim_i2t, targets)
    loss_t2i = F.cross_entropy(sim_t2i, targets)
    loss_itc = (loss_i2t + loss_t2i) / 2

    # ITM loss with improved negative sampling
    # Positive pair
    itm_output_pos = model(images, input_ids, attention_mask, mode="itm")

    if use_hard_negatives and batch_size > 1:
        # Hard negative sampling based on ITC similarity
        with torch.no_grad():
            # Find most similar incorrect pairs
            sim_matrix = sim_i2t.detach()
            # Mask diagonal (correct pairs)
            mask = torch.eye(batch_size, device=device).bool()
            sim_matrix.masked_fill_(mask, -float("inf"))
            # Get indices of most similar (hardest) negatives
            hard_neg_indices = sim_matrix.argmax(dim=1)
    else:
        # Fallback to random negative sampling if batch too small
        hard_neg_indices = torch.randperm(batch_size, device=device)

    # Create negative pairs using hard negatives
    input_ids_neg = input_ids[hard_neg_indices]
    attn_mask_neg = attention_mask[hard_neg_indices]
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
    val_loss_itc = 0
    val_loss_itm = 0
    val_loss_lm = 0

    with torch.no_grad():
        val_iter = itertools.islice(dataloader, num_val_steps)
        for i, batch in enumerate(val_iter):
            # Skip batch if None (filtered out due to duplicate images)
            if batch is None:
                continue

            images, input_ids, attention_mask = batch
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
            val_loss_itc += loss_itc.item()
            val_loss_itm += loss_itm.item()
            val_loss_lm += loss_lm.item()

    model.train()
    return (
        val_loss / num_val_steps,
        val_loss_itc / num_val_steps,
        val_loss_itm / num_val_steps,
        val_loss_lm / num_val_steps,
    )


def train(
    save_path="best_blip_model.pth",
    max_steps=500,
    batch_size=12,
    learning_rate=1e-5,
    weight_decay=0.05,
    val_interval=200,
    patience=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BLIP().to(device)

    # ITMヘッドだけ高い学習率を設定
    itm_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "itm_head" in name:
            itm_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
            {
                "params": itm_params,
                "lr": learning_rate * 10,
                "weight_decay": weight_decay,
            },  # 10倍の学習率
        ]
    )

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

    for i, batch in enumerate(train_iter):
        # Skip batch if None (filtered out due to duplicate images)
        if batch is None:
            continue

        images, input_ids, attention_mask = batch
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

        total_loss += loss.item()

        if i > 0 and i % val_interval == 0:
            current_train_loss = loss.item()
            current_val_loss, val_loss_itc, val_loss_itm, val_loss_lm = validate(
                model, val_loader, device
            )
            print(
                f"\nStep: {i} | Train Loss: {current_train_loss:.4f} | Val Loss: {current_val_loss:.4f}"
            )
            print(
                f"  Val ITC Loss: {val_loss_itc:.4f} | Val ITM Loss: {val_loss_itm:.4f} | Val LM Loss: {val_loss_lm:.4f}"
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
    # To create negative samples for ITM training, use a batch size of 2 or more.
    train(
        save_path="best_blip_model.pth",
        max_steps=30000,
        batch_size=16,
        weight_decay=0.05,
        val_interval=200,
        patience=3,
    )
