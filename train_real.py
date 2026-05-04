import os, json, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models
from PIL import Image
from datasets import load_dataset

# Config
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42
NUM_CLASSES = 10
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

TOP_CLASSES = [
    'Tshirts', 'Shirts', 'Casual Shoes', 'Watches', 'Sports Shoes',
    'Kurtas', 'Tops', 'Handbags', 'Heels', 'Sunglasses'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(TOP_CLASSES)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'].convert('RGB')
        label = self.class_to_idx[item['articleType']]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def build_model(num_classes=NUM_CLASSES):
    model = models.mobilenet_v3_small(weights='DEFAULT')
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'per_class': {TOP_CLASSES[i]: {
            'precision': round(per_prec[i], 4),
            'recall': round(per_rec[i], 4),
            'f1': round(per_f1[i], 4)
        } for i in range(len(TOP_CLASSES))}
    }, cm


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(TOP_CLASSES)), yticks=np.arange(len(TOP_CLASSES)),
           xticklabels=TOP_CLASSES, yticklabels=TOP_CLASSES,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix — Real Product Images (MobileNetV3-Small)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(len(TOP_CLASSES)):
        for j in range(len(TOP_CLASSES)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def export_onnx(model, save_path, device):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, save_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=18, do_constant_folding=True)
    print(f'ONNX exported: {save_path}')


def embed_onnx(onnx_path):
    from onnx import ModelProto, save as onnx_save
    model = ModelProto()
    with open(onnx_path, 'rb') as f:
        model.ParseFromString(f.read())
    for init in model.graph.initializer:
        init.data_location = 0
    onnx_save(model, onnx_path)
    print(f'Weights embedded in {onnx_path}')


def main():
    print('=' * 60)
    print('Retail Image Classifier — Real Product Images')
    print('MobileNetV3-Small + Transfer Learning')
    print('=' * 60)

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    start_time = time.time()

    # Load dataset
    print(f'\n[1/7] Loading Fashion Product Images...')
    full_ds = load_dataset("ashraq/fashion-product-images-small", split="train")

    # Filter to top 10 classes
    filtered_indices = [i for i, item in enumerate(full_ds)
                        if item['articleType'] in TOP_CLASSES]
    ds = Subset(ProductDataset(full_ds), filtered_indices)
    print(f'Filtered to {len(ds)} images (top 10 classes)')

    # Stratified split
    test_size = int(len(ds) * TEST_SPLIT)
    val_size = int(len(ds) * VAL_SPLIT)
    train_size = len(ds) - test_size - val_size

    train_ds, val_ds, test_ds = random_split(
        ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_ds = Subset(ds, train_ds.indices)
    val_ds = Subset(ds, val_ds.indices)
    test_ds = Subset(ds, test_ds.indices)

    # Create dataset objects with transforms
    train_ds_transformed = ProductDataset(
        Subset(full_ds, [filtered_indices[i] for i in train_ds.indices]),
        transform=get_transform(True)
    )
    val_ds_transformed = ProductDataset(
        Subset(full_ds, [filtered_indices[i] for i in val_ds.indices]),
        transform=get_transform(False)
    )
    test_ds_transformed = ProductDataset(
        Subset(full_ds, [filtered_indices[i] for i in test_ds.indices]),
        transform=get_transform(False)
    )

    print(f'Train: {len(train_ds_transformed)}, Val: {len(val_ds_transformed)}, Test: {len(test_ds_transformed)}')

    train_loader = DataLoader(train_ds_transformed, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds_transformed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds_transformed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Build model
    print('\n[2/7] Building MobileNetV3-Small with transfer learning...')
    model = build_model(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Train
    print('\n[3/7] Training...')
    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, 'best_model_real.pth')

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '
              f'Train: {train_loss:.4f}/{train_acc:.4f} | '
              f'Val: {val_loss:.4f}/{val_acc:.4f} | '
              f'Time: {epoch_time:.1f}s | LR: {lr:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  -> Best model saved! (val_acc: {val_acc:.4f})')

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time:.0f}s ({total_time/60:.1f} min)')
    print(f'Best val acc: {best_val_acc:.4f}')

    # Evaluate
    print('\n[4/7] Evaluating on test set...')
    model.load_state_dict(torch.load(best_path, weights_only=True))
    y_true, y_pred = predict(model, test_loader, device)
    metrics, cm = compute_metrics(y_true, y_pred)

    print(f'\nTest Accuracy:  {metrics["accuracy"]}')
    print(f'Test Precision: {metrics["precision"]}')
    print(f'Test Recall:    {metrics["recall"]}')
    print(f'Test F1:        {metrics["f1"]}')
    print('\nPer-class:')
    for cls, v in metrics['per_class'].items():
        print(f'  {cls:15s}: P={v["precision"]:.4f} R={v["recall"]:.4f} F1={v["f1"]:.4f}')

    # Save metrics
    with open(os.path.join(METRICS_DIR, 'metrics_real.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    cm_path = os.path.join(METRICS_DIR, 'confusion_matrix_real.png')
    plot_confusion_matrix(cm, cm_path)
    print(f'\nMetrics saved to {METRICS_DIR}/')

    # Export ONNX
    print('\n[5/7] Exporting to ONNX...')
    onnx_path = os.path.join(MODELS_DIR, 'retail_classifier_real.onnx')
    export_onnx(model, onnx_path, device)
    embed_onnx(onnx_path)

    size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f'Final ONNX size: {size:.2f} MB')

    print('\n[6/7] Summary')
    print('=' * 60)
    print(f'  Dataset:      Fashion Product Images (real photos)')
    print(f'  Classes:      {len(TOP_CLASSES)}: {", ".join(TOP_CLASSES[:5])}...')
    print(f'  Samples:      {len(train_ds_transformed)}/{len(val_ds_transformed)}/{len(test_ds_transformed)}')
    print(f'  Model:        MobileNetV3-Small + Transfer Learning')
    print(f'  Epochs:       {NUM_EPOCHS}')
    print(f'  Best Val Acc: {best_val_acc:.4f}')
    print(f'  Test Acc:     {metrics["accuracy"]}')
    print(f'  Test F1:      {metrics["f1"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()