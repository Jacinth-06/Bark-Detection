import os
import sys
import json
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "bark_cnn_model.pth"
CAMPUS_JSON_PATH = "Campus_Trees.json"
TEST_DATASET_DIR = "test_datset"
PORT = int(os.environ.get("PORT", 5001))

# Mapping from folder name to the exact key in Campus_Trees.json
FOLDER_TO_CAMPUS_NAME = {
    "alstonia_scholaris": "Alstonia scholaris",
    "azadirachta_indica": "Azadirachta indica",
    "bauhinia_purpurea": "Bauhinia purpurea",
    "calophyllum_inophyllum": "Calophyllum inophyllum",
    "ficus_benjamina": "Ficus benjamina",
    "lagerstroemia_speciosa": "Lagerstroemia speciosa",
    "millingtonia_hortensis": "Bignonia",
    "mimusops_elengi": "Mimusops elengi",
    "peltophorum_pterocarpum": "Peltophorum ferrugineum",
    "syzygium_cumini": "Eugenia jambolana",
    "tabebuia_rosea": "Tabebuia rosea",
}

COMMON_NAMES = {
    "Alstonia scholaris": "Devil's Tree (Saptaparni)",
    "Azadirachta indica": "Neem Tree",
    "Bauhinia purpurea": "Purple Orchid Tree",
    "Calophyllum inophyllum": "Alexandrian Laurel (Punnai)",
    "Ficus benjamina": "Weeping Fig",
    "Lagerstroemia speciosa": "Queen's Crape Myrtle (Pride of India)",
    "Bignonia": "Indian Cork Tree (Millingtonia hortensis)",
    "Mimusops elengi": "Spanish Cherry (Magizham)",
    "Peltophorum ferrugineum": "Yellow Flame Tree / Copperpod",
    "Eugenia jambolana": "Jamun (Black Plum)",
    "Tabebuia rosea": "Pink Trumpet Tree",
}

# 1. Global Model Loader
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model checkpoint '{MODEL_PATH}' not found!")
    print("Please make sure bark_cnn_model.pth is in the current directory.")
    sys.exit(1)

print(f"🧠 Loading model '{MODEL_PATH}' onto {device}...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['classes']

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()
print(f"✅ Model loaded with {len(classes)} classes.")

# Load Campus Metadata
campus_data = {}
if os.path.exists(CAMPUS_JSON_PATH):
    with open(CAMPUS_JSON_PATH, 'r') as f:
        campus_data = json.load(f)
    print(f"🏛️ Loaded campus records for {len(campus_data)} tree species.")

# Preprocessing Pipeline
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def run_inference(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.topk(probs, 3)

    pred_folder = classes[top_idx[0].item()]
    confidence = float(top_prob[0].item())

    campus_key = FOLDER_TO_CAMPUS_NAME.get(pred_folder, pred_folder)
    common_name = COMMON_NAMES.get(campus_key, "")
    tree_records = campus_data.get(campus_key, [])

    top3_list = []
    for rank in range(min(3, len(top_prob))):
        c_name = classes[top_idx[rank].item()]
        c_mapped = FOLDER_TO_CAMPUS_NAME.get(c_name, c_name)
        c_common = COMMON_NAMES.get(c_mapped, "")
        top3_list.append({
            "name": f"{c_mapped} ({c_common})" if c_common else c_mapped,
            "probability": float(top_prob[rank].item())
        })

    return {
        "species_name": campus_key,
        "common_name": common_name,
        "confidence": confidence,
        "top3": top3_list,
        "campus_records": tree_records
    }


# Try running with Flask if available, or seamlessly fallback to Python's built-in http.server
try:
    from flask import Flask, request, jsonify, send_from_directory, send_file
    
    app = Flask(__name__, static_folder="static")

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    @app.route("/<path:filename>")
    def serve_static(filename):
        return send_from_directory("static", filename)

    @app.route("/samples/<filename>")
    def serve_sample(filename):
        sample_path = os.path.join(TEST_DATASET_DIR, filename)
        if os.path.exists(sample_path):
            return send_file(sample_path)
        return jsonify({"error": "Sample not found"}), 404

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        try:
            image_bytes = file.read()
            results = run_inference(image_bytes)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def start_server():
        print(f"\n🌐 BarkVision Web App running on Flask!")
        print(f"👉 Open your browser at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop.\n")
        app.run(host="0.0.0.0", port=PORT, debug=False)

except ImportError:
    # Zero-dependency Fallback: Standard Library http.server
    import http.server
    import urllib.parse

    class BarkHTTPHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory="static", **kwargs)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.startswith("/samples/"):
                sample_name = urllib.parse.unquote(parsed.path[len("/samples/"):])
                sample_path = os.path.join(TEST_DATASET_DIR, sample_name)
                if os.path.exists(sample_path):
                    self.send_response(200)
                    if sample_name.lower().endswith(".webp"):
                        self.send_header("Content-Type", "image/webp")
                    elif sample_name.lower().endswith((".jpg", ".jpeg")):
                        self.send_header("Content-Type", "image/jpeg")
                    elif sample_name.lower().endswith(".png"):
                        self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(os.path.getsize(sample_path)))
                    self.end_headers()
                    with open(sample_path, "rb") as f:
                        self.wfile.write(f.read())
                    return
                else:
                    self.send_error(404, "Sample image not found")
                    return
            super().do_GET()

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/predict":
                content_type = self.headers.get("Content-Type", "")
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                try:
                    # Extract file bytes from multipart form
                    if "multipart/form-data" in content_type:
                        boundary = content_type.split("boundary=")[1].encode()
                        parts = body.split(b"--" + boundary)
                        image_bytes = None
                        for p in parts:
                            if b'name="file"' in p:
                                header_end = p.find(b"\r\n\r\n")
                                if header_end != -1:
                                    image_bytes = p[header_end + 4:].rstrip(b"\r\n")
                                    break
                    else:
                        image_bytes = body

                    if not image_bytes:
                        raise ValueError("No image data found in request")

                    results = run_inference(image_bytes)
                    res_json = json.dumps(results).encode("utf-8")

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(res_json)))
                    self.end_headers()
                    self.wfile.write(res_json)
                except Exception as e:
                    err_json = json.dumps({"error": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_json)))
                    self.end_headers()
                    self.wfile.write(err_json)
            else:
                self.send_error(404, "Endpoint not found")

    def start_server():
        print(f"\n🌐 BarkVision Web App running via Python HTTP Server!")
        print(f"👉 Open your browser at: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop.\n")
        server = http.server.HTTPServer(("0.0.0.0", PORT), BarkHTTPHandler)
        server.serve_forever()

if __name__ == "__main__":
    start_server()
