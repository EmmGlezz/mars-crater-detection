import gradio as gr
from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt')

def detect_craters(image):
    results = model(image)
    output = results[0].plot()
    return Image.fromarray(output)

demo = gr.Interface(
    fn=detect_craters,
    inputs=gr.Image(type="pil", label="Upload a Mars satellite image"),
    outputs=gr.Image(type="pil", label="Detected Craters"),
    title="Mars Crater Detection",
    description="Upload a Mars satellite image and the model will detect and highlight impact craters."
)

demo.launch()