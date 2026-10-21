# Tree Bark Recognition & Campus Tree Identifier

A Convolutional Neural Network (CNN) in PyTorch to identify tree bark species and cross-reference them with campus tree locations and VIP planters from `Campus_Trees.json`.

---

## 📁 Project Structure

```text
├── Campus_Trees.json           # Database of campus trees (locations, planters)
├── bark_data/                  # Dataset: ~3,500 images across 11 tree species
├── test_datset/                # Test sample images for demo
├── app.py                      # Web Application Server (Flask / Python HTTP)
├── static/                     # Web UI frontend (HTML, CSS, JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── train.py                    # PyTorch CNN training script (ResNet-18)
├── predict.py                  # CLI Prediction & Campus_Trees.json lookup
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🌲 Tree Species & Campus Mapping

| Folder in `bark_data` | Key in `Campus_Trees.json` | Common / Regional Name | Campus Trees |
| :--- | :--- | :--- | :--- |
| `alstonia_scholaris` | `Alstonia scholaris` | Devil's Tree (Saptaparni) | 4 |
| `azadirachta_indica` | `Azadirachta indica` | Neem Tree | 7 |
| `bauhinia_purpurea` | `Bauhinia purpurea` | Purple Orchid Tree | 3 |
| `calophyllum_inophyllum` | `Calophyllum inophyllum` | Alexandrian Laurel (Punnai) | 23 |
| `ficus_benjamina` | `Ficus benjamina` | Weeping Fig | 2 |
| `lagerstroemia_speciosa` | `Lagerstroemia speciosa` | Queen's Crape Myrtle | 6 |
| `millingtonia_hortensis` | `Bignonia` | Indian Cork Tree | 3 |
| `mimusops_elengi` | `Mimusops elengi` | Spanish Cherry (Magizham) | 13 |
| `peltophorum_pterocarpum`| `Peltophorum ferrugineum` | Yellow Flame / Copperpod | 25 |
| `syzygium_cumini` | `Eugenia jambolana` | Jamun (Black Plum) | 1 |
| `tabebuia_rosea` | `Tabebuia rosea` | Pink Trumpet Tree | 1 |

---

## 🚀 Setup & Installation

### 1. Install Requirements
Open your terminal inside this project directory:
```bash
pip install -r requirements.txt
```

---

## ⚡ Training the CNN Model

Run `train.py` to train the ResNet-18 CNN on the ~3,500 images:

```bash
python train.py
```

### What happens during training:
- **Device**: Automatically uses Apple Silicon GPU (`mps`) on macOS or CUDA/CPU.
- **Split**: 80% training (~2,830 images) and 20% validation (~700 images).
- **Augmentation**: Random horizontal flips, rotations (15°), and color jitter to prevent overfitting.
- **Output**: Saves the best-performing model checkpoint to `bark_cnn_model.pth`.

---

## 🔍 Making Predictions & Campus Lookup

To classify any bark photo and retrieve campus locations:

```bash
python predict.py "path/to/bark_image.jpg"
```

### Example:
```bash
python predict.py "bark_data/azadirachta_indica/47413484_bark.jpg"
```

### Output:
```text
=================================================================
🌲 BARK IDENTIFICATION & CAMPUS TREE LOOKUP
=================================================================
📸 Image Analyzed    : bark_data/azadirachta_indica/47413484_bark.jpg
🌿 Botanical Name    : Azadirachta indica
🍃 Common Name       : Neem Tree
🎯 Prediction Score  : 96.20%
📍 Campus Inventory  : 7 trees recorded in campus database
-----------------------------------------------------------------
🏛️  Known Campus Locations & Planters:
   [1] Location: MH-Near Q block
       Planted by: Mr. Karthi - South Indian Actor
   [2] Location: MH-Near Q Block
       Planted by: Prof.B.V.R.Chowdari - Senior Executive Director, Materials Research Society of Singapore
   ...
-----------------------------------------------------------------
📊 Top 3 Predictions:
   1. Azadirachta indica: 96.20%
   2. Calophyllum inophyllum: 2.15%
   3. Mimusops elengi: 0.85%
=================================================================
```
# Bark-Detection
