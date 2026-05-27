import os
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0

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

# === Preprocessing with optional noise ===
def distort(image, simulate=True):
    if simulate:
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.uint8)

        if np.random.rand() < 0.5:
            arr = cv2.GaussianBlur(arr, (5, 5), 0)

        if np.random.rand() < 0.5:
            _, arr = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            arr = cv2.imdecode(arr, 1)

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(arr)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)

# === Evaluation with Accuracy ===
def evaluate(folder="realworld_samples/", simulate_noise=True):
    correct = 0
    total = 0

    for label_name in ["real", "fake"]:
        subfolder = os.path.join(folder, label_name)

        if not os.path.exists(subfolder):
            print(f"⚠️ Missing folder: {subfolder}")
            continue

        for file in os.listdir(subfolder):
            path = os.path.join(subfolder, file)

            if not os.path.isfile(path):
                continue

            try:
                # === IMAGE ===
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image = Image.open(path).convert("RGB")
                    tensor = distort(image, simulate=simulate_noise)

                # === VIDEO ===
                elif file.lower().endswith((".mp4", ".mov")):
                    cap = cv2.VideoCapture(path)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        print(f"{file}: ❌ Error reading video")
                        continue

                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    tensor = distort(image, simulate=simulate_noise)

                else:
                    print(f"{file}: Unsupported type")
                    continue

                # === Prediction ===
                with torch.no_grad():
                    out = model(tensor)
                    prob = torch.softmax(out, dim=1)[0]
                    conf, pred = torch.max(prob, dim=0)

                predicted_label = "real" if pred.item() == 0 else "fake"

                # === Accuracy Calculation ===
                if predicted_label == label_name:
                    correct += 1

                total += 1

                print(f"{file:<25} ➤ Pred: {predicted_label:<5} | Actual: {label_name:<5} ({conf.item()*100:.2f}%)")

            except Exception as e:
                print(f"{file}: ⚠️ {e}")

    # === Final Accuracy ===
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n🎯 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("❌ No valid samples found.")

# === Run ===
if __name__ == "__main__":
    evaluate(simulate_noise=True)