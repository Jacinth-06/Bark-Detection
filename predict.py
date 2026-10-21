import sys
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "bark_cnn_model.pth"
CAMPUS_JSON_PATH = "Campus_Trees.json"

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

def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model checkpoint '{MODEL_PATH}' not found!")
        print("Please run 'python train.py' first to train and save the model.")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    return model, classes

def predict(image_path):
    # Check direct path first, then look inside common subfolders (e.g. test_datset)
    if not os.path.exists(image_path):
        candidate = os.path.join("test_datset", image_path)
        if os.path.exists(candidate):
            image_path = candidate
        else:
            # Try searching in test_datset for partial match
            found = False
            if os.path.exists("test_datset"):
                for fname in os.listdir("test_datset"):
                    if fname.lower() == image_path.lower() or fname.lower().startswith(image_path.lower()):
                        image_path = os.path.join("test_datset", fname)
                        found = True
                        break
            if not found:
                print(f"❌ Error: Image '{image_path}' not found!")
                print("💡 Tip: Provide the path from your current folder, for example:")
                print("   python predict.py test_datset/Neem-3-2.png.webp")
                sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, classes = load_model(device)

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.topk(probs, 3)

    pred_folder = classes[top_idx[0].item()]
    confidence = top_prob[0].item()

    # Map to Campus_Trees.json
    campus_key = FOLDER_TO_CAMPUS_NAME.get(pred_folder, pred_folder)
    common_name = COMMON_NAMES.get(campus_key, "")

    # Load Campus metadata
    campus_records = []
    if os.path.exists(CAMPUS_JSON_PATH):
        with open(CAMPUS_JSON_PATH, 'r') as f:
            campus_data = json.load(f)
            campus_records = campus_data.get(campus_key, [])

    # Display Results
    print("\n" + "=" * 65)
    print("🌲 BARK IDENTIFICATION & CAMPUS TREE LOOKUP")
    print("=" * 65)
    print(f"📸 Image Analyzed    : {image_path}")
    print(f"🌿 Botanical Name    : {campus_key}")
    if common_name:
        print(f"🍃 Common Name       : {common_name}")
    print(f"🎯 Prediction Score  : {confidence * 100:.2f}%")
    print(f"📍 Campus Inventory  : {len(campus_records)} trees recorded in campus database")
    print("-" * 65)

    if campus_records:
        print("🏛️  Known Campus Locations & Planters:")
        # Show top 5 locations
        for i, item in enumerate(campus_records[:5], 1):
            loc = item.get("location", "Campus Grounds")
            planter = item.get("planter", "N/A")
            desig = item.get("designation", "")
            desig_str = f" - {desig}" if desig else ""
            print(f"   [{i}] Location: {loc}")
            print(f"       Planted by: {planter}{desig_str}")
        if len(campus_records) > 5:
            print(f"       ... and {len(campus_records) - 5} more trees on campus.")
    else:
        print("ℹ️  No campus tree locations recorded for this species.")

    # Show top 3 probabilities
    print("-" * 65)
    print("📊 Top 3 Predictions:")
    for rank in range(min(3, len(top_prob))):
        c_name = classes[top_idx[rank].item()]
        mapped_name = FOLDER_TO_CAMPUS_NAME.get(c_name, c_name)
        prob = top_prob[rank].item()
        print(f"   {rank+1}. {mapped_name}: {prob*100:.2f}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py bark_data/azadirachta_indica/some_image.jpg")
    else:
        predict(sys.argv[1])
