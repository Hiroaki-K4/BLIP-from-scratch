import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataloader(tokenizer, batch_size=8, split="train"):
    if split == "train":
        # Random crop
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
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
        image_ids = []  # Track image IDs

        for item in batch:
            images.append(transform(item["image"].convert("RGB")))
            captions.append(item["caption"])
            # Get image ID (for COCO dataset)
            image_ids.append(
                item.get("image_id", id(item["image"]))
            )  # fallback to object id

        # Check for duplicate images within batch
        unique_ids = set()
        filtered_images = []
        filtered_captions = []

        for i, img_id in enumerate(image_ids):
            if img_id not in unique_ids:
                unique_ids.add(img_id)
                filtered_images.append(images[i])
                filtered_captions.append(captions[i])

        # Skip batch if too few samples after deduplication
        if len(filtered_images) < 2:
            return None  # Skip this batch

        tokenized = tokenizer(
            filtered_captions,
            padding=True,
            truncation=True,
            max_length=30,
            return_tensors="pt",
        )
        return (
            torch.stack(filtered_images),
            tokenized["input_ids"],
            tokenized["attention_mask"],
        )

    dataset = dataset.shuffle(seed=42, buffer_size=1000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,  # Exclude incomplete batches
    )
    return loader
