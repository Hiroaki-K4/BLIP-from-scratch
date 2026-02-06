import itertools

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

from blip_itc import BLIP_ITC_MODEL
from dataloader import get_dataloader


def train(max_steps=500, batch_size=12, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Max Steps: {max_steps}")
    model = BLIP_ITC_MODEL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataloader = get_dataloader(tokenizer=tokenizer, batch_size=batch_size)
    train_iter = itertools.islice(dataloader, max_steps)

    model.train()
    pbar = tqdm(enumerate(train_iter), total=max_steps, desc="Training")

    total_loss = 0
    for i, (images, input_ids, attention_mask) in pbar:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward
        sim_i2t, sim_t2i = model(images, input_ids, attention_mask)

        targets = torch.arange(images.size(0)).to(device)

        loss_i2t = F.cross_entropy(sim_i2t, targets)
        loss_t2i = F.cross_entropy(sim_t2i, targets)
        loss = (loss_i2t + loss_t2i) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix(
            {"loss": f"{current_loss:.4f}", "avg_loss": f"{total_loss/(i+1):.4f}"}
        )

    save_path = f"blip_itc_steps{max_steps}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining finished. Model saved to {save_path}")


if __name__ == "__main__":
    train(max_steps=1000, batch_size=32)
