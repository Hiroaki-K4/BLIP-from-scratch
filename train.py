import itertools

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from blip_itc import BLIP_ITC_MODEL
from dataloader import get_dataloader


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
            sim_i2t, sim_t2i = model(images, input_ids, attention_mask)
            targets = torch.arange(images.size(0)).to(device)

            loss_i2t = F.cross_entropy(sim_i2t, targets)
            loss_t2i = F.cross_entropy(sim_t2i, targets)
            val_loss += (loss_i2t + loss_t2i) / 2

    model.train()
    return (val_loss / num_val_steps).item()


def train(
    save_path="best_blip_itc_model.pth",
    max_steps=500,
    batch_size=12,
    learning_rate=1e-4,
    val_interval=200,
    patience=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BLIP_ITC_MODEL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

        # Forward & Update
        sim_i2t, sim_t2i = model(images, input_ids, attention_mask)
        targets = torch.arange(images.size(0)).to(device)
        loss = (
            F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)
        ) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i > 0 and i % val_interval == 0:
            avg_train_loss = total_loss / (i + 1)
            current_val_loss = validate(model, val_loader, device)

            print(
                f"\nStep: {i} | Train Loss: {avg_train_loss:.4f} | Val Loss: {current_val_loss:.4f}"
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
        save_path="best_blip_itc_model.pth",
        max_steps=10000,
        batch_size=32,
        val_interval=100,
        patience=2,
    )
