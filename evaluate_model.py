import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import evaluate
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# 📁 Paths
DATA_PATH = "data/flickr30k/flickr30k_data.json"
FINETUNED_PATH = "finetuned_model"

# ✅ Load model and processor
processor = BlipProcessor.from_pretrained(FINETUNED_PATH)
model = BlipForConditionalGeneration.from_pretrained(FINETUNED_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 📥 Load dataset
with open(DATA_PATH, "r") as f:
    data = json.load(f)

# Select test samples: 6000-7000
test_samples = data[6000:7000]

# Initialize metrics
smooth_fn = SmoothingFunction().method4
bleu_scores = []
references = []
predictions = []

# 🧠 Inference
for item in tqdm(test_samples):
    image_path = item["image"]
    ground_truth = item["caption"]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        continue

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    predicted_caption = processor.decode(output_ids[0], skip_special_tokens=True)

    # BLEU
    bleu = sentence_bleu([ground_truth.split()], predicted_caption.split(), smoothing_function=smooth_fn)
    bleu_scores.append(bleu)

    # ROUGE
    references.append(ground_truth)
    predictions.append(predicted_caption)

# Compute scores
avg_bleu = sum(bleu_scores) / len(bleu_scores)
rouge_output = rouge.compute(predictions=predictions, references=references)

avg_bleu, rouge_output
# Compute scores
avg_bleu = sum(bleu_scores) / len(bleu_scores)
rouge_output = rouge.compute(predictions=predictions, references=references)

# Print final scores
print(f"\n🟦 Average BLEU score on 1000 test images: {avg_bleu:.4f}")
print("🟥 ROUGE scores:")
for k, v in rouge_output.items():
    print(f"  {k}: {v:.4f}")

with open("blip_test_output.json", "w") as f:
    json.dump([
        {"image": item["image"], "caption": item["caption"], "prediction": pred, "bleu": score}
        for item, pred, score in zip(test_samples, predictions, bleu_scores)
    ], f, indent=2)
print("📁 Results saved to blip_test_output.json")

import matplotlib.pyplot as plt
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate

# Load data
with open("data/flickr30k/flickr30k_data.json", "r") as f:
    data = json.load(f)

with open("blip_test_output.json", "r") as f:
    predictions_data = json.load(f)

# Use first 10 samples
samples = predictions_data[:10]

# Prepare metrics
rouge = evaluate.load("rouge")
smooth_fn = SmoothingFunction().method4

bleu_scores = []
rouge_l_scores = []

# Calculate BLEU and ROUGE-L
for item in samples:
    reference = item["caption"]
    prediction = item["prediction"]

    # BLEU
    bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth_fn)
    bleu_scores.append(bleu)

    # ROUGE-L (we get all rouge metrics but only extract rougeL)
    rouge_out = rouge.compute(predictions=[prediction], references=[reference])
    rouge_l_scores.append(rouge_out["rougeL"])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), bleu_scores, marker='o', label="BLEU")
plt.plot(range(1, 11), rouge_l_scores, marker='s', label="ROUGE-L")
plt.title("BLEU & ROUGE-L Scores for First 10 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bleu_rouge_first10.png")
plt.show()
