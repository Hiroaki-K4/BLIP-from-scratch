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

    # ITM loss - probabilistic negative sampling (1:1 ratio for balance)
    # Positive pairs
    itm_output_pos = model(images, input_ids, attention_mask, mode="itm")

    # Probabilistic negative sampling based on ITC similarity
    with torch.no_grad():
        # Create probability distribution from similarity (higher similarity = higher probability)
        weights_i2t = F.softmax(sim_i2t[:, :batch_size], dim=1) + 1e-4
        weights_i2t.fill_diagonal_(0)  # Don't sample yourself

    # Select negative text for each image (1:1 ratio)
    text_neg_indices = []
    for b in range(batch_size):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_neg_indices.append(neg_idx)
    text_neg_indices = torch.tensor(text_neg_indices, device=device)

    # Generate negative pairs (correct image with wrong text)
    input_ids_neg = input_ids[text_neg_indices]
    attn_mask_neg = attention_mask[text_neg_indices]
    itm_output_neg = model(images, input_ids_neg, attn_mask_neg, mode="itm")

    # Combine: positive + negative (ratio 1:1 for balanced learning)
    itm_logits = torch.cat([itm_output_pos, itm_output_neg], dim=0)
    itm_labels = torch.cat(
        [
            torch.ones(batch_size, dtype=torch.long),  # Positive
            torch.zeros(batch_size, dtype=torch.long),  # Negative
        ],
        dim=0,
    ).to(device)
    loss_itm = F.cross_entropy(itm_logits, itm_labels)

    # Debug: Calculate ITM accuracy and prediction stats
    with torch.no_grad():
        itm_probs = F.softmax(itm_logits, dim=1)
        itm_preds = itm_logits.argmax(dim=1)
        itm_acc = (itm_preds == itm_labels).float().mean()

        # Accuracy for positive and negative separately
        pos_acc = (itm_preds[:batch_size] == 1).float().mean()
        neg_acc = (itm_preds[batch_size:] == 0).float().mean()

        # Average confidence for match class
        match_confidence = itm_probs[:, 1].mean()

    # Return debug info as well
    itm_debug = {
        "acc": itm_acc.item(),
        "pos_acc": pos_acc.item(),
        "neg_acc": neg_acc.item(),
        "match_conf": match_confidence.item(),
    }

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

    return loss_itc, loss_itm, loss_lm, itm_debug


def validate(model, tokenizer, batch_size, device, num_val_steps=50):
    """Calculate loss for validation data - creates fresh dataloader each time"""
    model.eval()
    val_loss = 0
    val_loss_itc = 0
    val_loss_itm = 0
    val_loss_lm = 0

    # Create fresh validation dataloader to avoid data exhaustion
    val_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="validation"
    )

    with torch.no_grad():
        val_iter = itertools.islice(val_loader, num_val_steps)
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
            loss_itc, loss_itm, loss_lm, _ = compute_losses(
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
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.044,
    val_interval=500,
    patience=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BLIP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="train"
    )
    val_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="validation"
    )

    # Variables for Early Stopping
    best_val_loss = float("inf")
    counter = 0
    step = 0  # Track actual training steps

    model.train()
    total_loss = 0

    print(f"Starting training for max {max_steps} steps...")

    # Loop infinitely through the dataset
    while step < max_steps:
        for i, batch in enumerate(train_loader):
            # Skip batch if None (filtered out due to duplicate images)
            if batch is None:
                continue

            # Check if we've reached max_steps
            if step >= max_steps:
                print(f"Reached max_steps ({max_steps}). Training finished.")
                break

            images, input_ids, attention_mask = batch
            images, input_ids, attention_mask = (
                images.to(device),
                input_ids.to(device),
                attention_mask.to(device),
            )

            # Compute all losses using the shared function
            loss_itc, loss_itm, loss_lm, itm_debug = compute_losses(
                model, images, input_ids, attention_mask, device
            )

            # Total loss: combine all three losses
            loss = loss_itc + loss_itm + loss_lm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step > 0 and step % val_interval == 0:
                current_train_loss = loss.item()
                current_val_loss, val_loss_itc, val_loss_itm, val_loss_lm = validate(
                    model, tokenizer, batch_size, device
                )
                print(
                    f"\nStep: {step} | Train Loss: {current_train_loss:.4f} | Val Loss: {current_val_loss:.4f}"
                )
                print(
                    f"  Val ITC Loss: {val_loss_itc:.4f} | Val ITM Loss: {val_loss_itm:.4f} | Val LM Loss: {val_loss_lm:.4f}"
                )
                print(
                    f"  ITM Debug - Acc: {itm_debug['acc']:.3f} | Pos Acc: {itm_debug['pos_acc']:.3f} | Neg Acc: {itm_debug['neg_acc']:.3f} | Match Conf: {itm_debug['match_conf']:.3f}"
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
                    return  # Exit the function completely

        # Check if we broke out of inner loop due to max_steps
        if step >= max_steps:
            break

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Final training steps: {step}/{max_steps}")


if __name__ == "__main__":
    # To create negative samples for ITM training, use a batch size of 2 or more.
    train(
        save_path="best_blip_model.pth",
        max_steps=300000,
        batch_size=16,
        weight_decay=0.044,
        val_interval=1000,
        patience=3,
    )
