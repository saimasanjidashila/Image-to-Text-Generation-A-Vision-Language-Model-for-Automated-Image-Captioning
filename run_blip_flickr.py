from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import os

# 📁 Load Flickr30k data
DATA_PATH = "data/flickr30k/flickr30k_data.json"

with open(DATA_PATH, "r") as f:
    data = json.load(f)

# 🧹 Dataset class
class Flickr30kDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        caption = item["caption"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding

# ✅ Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 📦 Split dataset
train_data = Flickr30kDataset(data[:5000], processor)
eval_data = Flickr30kDataset(data[-1000:], processor)

# 🛠️ Training setup
training_args = TrainingArguments(
    output_dir="./blip_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none"
)

# ⚙️ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=processor,
)

# 🚀 Train
trainer.train()

# 💾 Save model and processor
output_dir = "finetuned_model"
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"✅ Model saved to {output_dir}")
