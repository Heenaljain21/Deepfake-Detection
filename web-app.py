import gradio as gr
import torch
import mimetypes
from PIL import Image
import cv2
from torchvision.models import efficientnet_b0
from torchvision import transforms

# === Load Model (FIXED) ===
def load_model():
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    state_dict = torch.load("models/best_model-v3.pt", map_location="cpu")
    
    # Handle Lightning checkpoints
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Remove unwanted prefixes like "model."
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Inference Logic ===
def predict_file(file_obj):
    if file_obj is None:
        return "⚠️ No file selected", "", None

    path = file_obj.name
    mime, _ = mimetypes.guess_type(path)

    # === IMAGE ===
    if mime and mime.startswith("image"):
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        label = "🟢 Real" if pred.item() == 0 else "🔴 Deepfake"
        return label, f"{conf.item()*100:.2f}%", img

    # === VIDEO ===
    elif mime and mime.startswith("video"):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "❌ Error reading video", "", None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        label = "🟢 Real (1st frame)" if pred.item() == 0 else "🔴 Deepfake (1st frame)"
        return label, f"{conf.item()*100:.2f}%", img

    else:
        return "❌ Unsupported file type", "", None

# === Gradio UI ===
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("## 🧠 Deepfake Detector\nUpload an image or video to detect if it's real or fake.")

    file_input = gr.File(
        label="Upload Image or Video",
        file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
    )

    with gr.Row():
        prediction = gr.Textbox(label="Prediction", interactive=False)
        confidence = gr.Textbox(label="Confidence (%)", interactive=False)

    preview = gr.Image(label="Preview", interactive=False)

    file_input.change(
        fn=predict_file,
        inputs=file_input,
        outputs=[prediction, confidence, preview]
    )

# Launch app
demo.launch()