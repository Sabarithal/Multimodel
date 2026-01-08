import os
import uuid
import gradio as gr
import whisper
import torch
from transformers import pipeline
from gtts import gTTS
from PIL import Image

print("Starting app...")

# -----------------------------
# Performance & Memory Fixes
# -----------------------------
torch.set_num_threads(1)   # reduce CPU/RAM pressure

# -----------------------------
# Paths
# -----------------------------
FFMPEG_PATH = r"C:\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin"
INPUT_AUDIO = "inputs/audio.mp3"
INPUT_IMAGE = "inputs/image.jpeg"

os.environ["PATH"] += os.pathsep + FFMPEG_PATH
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Lazy-loaded models (IMPORTANT)
# -----------------------------
speech_model = None
image_to_text = None
llm = None

def load_whisper():
    global speech_model
    if speech_model is None:
        print("Loading Whisper (tiny)...")
        speech_model = whisper.load_model("tiny")

def load_image_caption():
    global image_to_text
    if image_to_text is None:
        print("Loading Image Caption model (lightweight)...")
        image_to_text = pipeline(
            "image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            device=-1  # force CPU
        )

def load_llm():
    global llm
    if llm is None:
        print("Loading LLM (gpt2)...")
        llm = pipeline(
            "text-generation",
            model="gpt2",
            device=-1
        )

# -----------------------------
# Main function
# -----------------------------
def run_multimodal(user_text):
    print("Processing inputs...")

    speech_text = ""
    image_caption = ""

    # 🎤 Audio
    if os.path.exists(INPUT_AUDIO):
        load_whisper()
        speech_text = speech_model.transcribe(INPUT_AUDIO)["text"]

    # 🖼️ Image
    if os.path.exists(INPUT_IMAGE):
        load_image_caption()
        img = Image.open(INPUT_IMAGE).convert("RGB")
        image_caption = image_to_text(img)[0]["generated_text"]

    # 🧠 LLM
    load_llm()

    prompt = f"""
Speech Input: {speech_text}
Image Description: {image_caption}
User Text: {user_text}
Answer clearly and concisely.
"""

    result = llm(prompt, max_length=120, do_sample=True)[0]["generated_text"]

    # 🔊 Text-to-Speech
    out_audio = f"outputs/out_{uuid.uuid4().hex}.mp3"
    gTTS(result).save(out_audio)

    return result, out_audio

# -----------------------------
# Gradio UI
# -----------------------------
interface = gr.Interface(
    fn=run_multimodal,
    inputs=gr.Textbox(label="Your Text Prompt"),
    outputs=[
        gr.Textbox(label="AI Response"),
        gr.Audio(type="filepath", label="AI Voice")
    ],
    title="Multimodal AI (Low Memory Optimized)",
    description="Uses inputs/audio.mp3 and inputs/image.jpeg"
)

print("🚀 Launching Gradio app...")

interface.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False
)
