import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = "bark_data"
MODEL_SAVE_PATH = "bark_cnn_model.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def main():
    # 1. Device selection (Apple Silicon MPS, NVIDIA CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f" Using device: {device}")

    # 2. Image Transformations & Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    print(f" Loading dataset from '{DATA_DIR}'...")
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    total_imgs = len(full_dataset)
    print(f" Found {total_imgs} images across {num_classes} tree species:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")

    # 4. Train / Validation Split (80% train, 20% validation)
    train_size = int(0.8 * total_imgs)
    val_size = total_imgs - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val_transform to validation split
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\n Split: {train_size} training images, {val_size} validation images.")

    # 5. CNN Model (Pretrained ResNet-18)
    print(" Initializing CNN model (ResNet-18)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final classification layer for our 11 species
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print("\n⚡ Starting Training...\n" + "-" * 60)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = train_correct / train_size

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_correct / val_size

        print(f"Epoch [{epoch:2d}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.2f}%")

        # Save best model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'classes': classes,
                'val_acc': best_val_acc
            }, MODEL_SAVE_PATH)

    print("-" * 60)
    print(f"🎉 Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"💾 Model weights saved to '{MODEL_SAVE_PATH}'\n")

if __name__ == "__main__":
    main()
