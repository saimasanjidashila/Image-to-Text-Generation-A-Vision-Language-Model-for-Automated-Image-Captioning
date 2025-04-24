import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model
processor = BlipProcessor.from_pretrained("finetuned_model")
model = BlipForConditionalGeneration.from_pretrained("finetuned_model")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Inference function
def caption_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    return processor.decode(output_ids[0], skip_special_tokens=True)

# Launch UI
with gr.Blocks(css="""
footer {display: none !important;}
h1 {text-align: center !important;}
.upload-buttons > button:nth-child(2),
.upload-buttons > button:nth-child(3) {
    display: none !important;
}
""") as demo:
    gr.Markdown("# 📸 BLIP Image Captioning")
    gr.HTML("<p style='text-align: center;'>Upload an image and get a caption generated using the <b>fine-tuned BLIP</b> model.</p>")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil", sources=["upload"])
            generate_btn = gr.Button("🪄 Generate Caption")

        with gr.Column():
            caption_output = gr.Textbox(label="Generated Caption", lines=3)
            placeholder="Your caption will appear here...",
    generate_btn.click(fn=caption_image, inputs=image_input, outputs=caption_output)

demo.launch()
