from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

image = Image.open("original/caption.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs, max_length=50)
print("BLIP output:", processor.decode(out[0], skip_special_tokens=True))
