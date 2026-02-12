import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataloader(tokenizer, batch_size=8, split="train"):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = load_dataset("jxie/coco_captions", split=split, streaming=True)

    def collate_fn(batch):
        images = []
        captions = []
        for item in batch:
            images.append(transform(item["image"].convert("RGB")))
            captions.append(item["caption"])

        tokenized = tokenizer(
            captions, padding=True, truncation=True, max_length=30, return_tensors="pt"
        )
        return torch.stack(images), tokenized["input_ids"], tokenized["attention_mask"]

    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return loader
