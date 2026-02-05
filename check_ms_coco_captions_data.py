import textwrap

import matplotlib.pyplot as plt
from datasets import load_dataset

dataset = load_dataset("jxie/coco_captions", streaming=True)

num_images_to_show = 10
data_iter = iter(dataset["train"])

for i in range(num_images_to_show):
    example = next(data_iter)

    image = example["image"]
    caption = example["caption"]

    wrapped_caption = "\n".join(textwrap.wrap(caption, width=50))

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(wrapped_caption)
    plt.axis("off")
    plt.show()
