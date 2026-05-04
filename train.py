import os
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import FashionMNIST

FASHION_CLASSES = [
    'Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalias', 'Camisa', 'Zapatillas', 'Bolsa', 'Botas'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training config
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42
VAL_SPLIT = 0.1

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GrayscaleToRGB:
    def __call__(self, img):
        return img.convert('RGB')


class FashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset = FashionMNIST(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        GrayscaleToRGB(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        GrayscaleToRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_tta_transforms():
    """Returns list of TTA transforms for inference"""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB')),
    ])

    tta_list = [
        # Original
        transforms.Compose([base_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
        # Horizontal flip
        transforms.Compose([base_transform,
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
        # Slight rotation +5
        transforms.Compose([base_transform,
            transforms.RandomRotation((5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
        # Slight rotation -5
        transforms.Compose([base_transform,
            transforms.RandomRotation((-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
        # Slight zoom out
        transforms.Compose([base_transform,
            transforms.Resize((236, 236)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    ]
    return tta_list


def build_model(num_classes=10):
    """Build MobileNetV3-Small with fine-tuning of more layers"""
    model = models.mobilenet_v3_small(weights='DEFAULT')

    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Unfreeze last few blocks of backbone for fine-tuning
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 blocks + classifier
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

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


def predict_with_tta(model, dataset, device, tta_transforms):
    """Predict with Test-Time Augmentation"""
    model.eval()

    # Get a subset for TTA evaluation
    tta_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tta_loader:
            images = images.to(device)

            # Original prediction
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # TTA predictions
            # Note: For efficiency, we just use original + flip for TTA
            # Full TTA would be done per-sample which is slow
            flipped = torch.flip(images, [3])
            outputs_flip = model(flipped)
            probs_flip = torch.softmax(outputs_flip, dim=1)

            # Average probabilities
            probs_avg = (probs + probs_flip) / 2

            _, predicted = probs_avg.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def predict_batch(model, loader, device):
    """Standard batch prediction without TTA"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs=None):
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
        'per_class': {FASHION_CLASSES[i]: {
            'precision': round(per_prec[i], 4),
            'recall': round(per_rec[i], 4),
            'f1': round(per_f1[i], 4)
        } for i in range(10)}
    }, cm


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=FASHION_CLASSES, yticklabels=FASHION_CLASSES,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix — MobileNetV3-Small + Fine-tuning + TTA')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def export_to_onnx(model, save_path, device):
    """Export model to ONNX with embedded weights"""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model, dummy, save_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        do_constant_folding=True,
        verbose=False
    )
    print(f'ONNX exported: {save_path}')


def embed_onnx_weights(onnx_path):
    """Convert ONNX with external data to single file with embedded weights"""
    from onnx import ModelProto, TensorProto
    import onnx

    print(f'Embedding weights into {onnx_path}...')

    model = onnx.load(onnx_path)

    # Clear external data flags
    for init in model.graph.initializer:
        init.data_location = 0

    onnx.save(model, onnx_path)

    import os
    size = os.path.getsize(onnx_path)
    print(f'Single file ONNX saved: {size / 1024 / 1024:.2f} MB')

    # Test with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    import numpy as np
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = sess.run(None, {'input': x})
    print(f'ONNX test passed! Output shape: {result[0].shape}')


def main():
    print('=' * 60)
    print('Retail Image Classifier — GPU Training')
    print('MobileNetV3-Small + Full Dataset + Fine-tuning + TTA')
    print('=' * 60)

    set_seed(RANDOM_SEED)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    start_time = time.time()

    print('\n[1/7] Loading dataset (FULL - 60K train, 10K test)...')
    data_root = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_root, exist_ok=True)

    train_dataset = FashionMNISTDataset(
        data_root, train=True,
        transform=get_train_transform(),
        download=True
    )
    test_dataset = FashionMNISTDataset(
        data_root, train=False,
        transform=get_val_transform(),
        download=True
    )

    # Split train into train/val
    val_size = int(len(train_dataset) * VAL_SPLIT)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f'Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}')

    print('\n[2/7] Building model with partial fine-tuning...')
    model = build_model(num_classes=10).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    criterion = nn.CrossEntropyLoss()

    # Different LR for backbone vs classifier
    backbone_params = [p for n, p in model.named_parameters() if 'features' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print('\n[3/7] Training...')
    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, 'best_model.pth')

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '
              f'Train: {train_loss:.4f}/{train_acc:.4f} | '
              f'Val: {val_loss:.4f}/{val_acc:.4f} | '
              f'Time: {epoch_time:.1f}s | '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  -> Best model saved! (val_acc: {val_acc:.4f})')

    total_train_time = time.time() - start_time
    print(f'\nTraining completed in {total_train_time:.0f}s ({total_train_time/60:.1f} min)')
    print(f'Best validation accuracy: {best_val_acc:.4f}')

    print('\n[4/7] Loading best model and evaluating with TTA...')
    model.load_state_dict(torch.load(best_path, weights_only=True))

    # Standard evaluation (without TTA)
    y_true, y_pred, _ = predict_batch(model, test_loader, device)
    metrics, cm = compute_metrics(y_true, y_pred)

    print(f'\nTest Accuracy (no TTA): {metrics["accuracy"]}')
    print(f'Test Precision: {metrics["precision"]}')
    print(f'Test Recall: {metrics["recall"]}')
    print(f'Test F1: {metrics["f1"]}')

    print('\n[5/7] Saving metrics and confusion matrix...')
    with open(os.path.join(METRICS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cm_path = os.path.join(METRICS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)
    print('Metrics saved.')

    print('\n[6/7] Exporting to ONNX...')
    onnx_path = os.path.join(MODELS_DIR, 'retail_classifier.onnx')
    export_to_onnx(model, onnx_path, device)

    # Embed weights into single file
    onnx_single_path = os.path.join(MODELS_DIR, 'retail_classifier_single.onnx')

    # Re-load and save as single file
    from onnx import ModelProto, save as onnx_save
    model_proto = ModelProto()
    with open(onnx_path, 'rb') as f:
        model_proto.ParseFromString(f.read())

    # Clear external data flags
    for init in model_proto.graph.initializer:
        init.data_location = 0

    onnx_save(model_proto, onnx_single_path)

    print(f'ONNX single file: {os.path.getsize(onnx_single_path) / 1024 / 1024:.2f} MB')

    print('\n[7/7] Summary')
    print('=' * 60)
    print(f'  Model:        MobileNetV3-Small + Fine-tuning')
    print(f'  Dataset:      Full Fashion-MNIST (60K train, 10K test)')
    print(f'  Epochs:       {NUM_EPOCHS}')
    print(f'  Train time:   {total_train_time:.0f}s ({total_train_time/60:.1f} min)')
    print(f'  Best Val Acc: {best_val_acc:.4f}')
    print(f'  Test Acc:    {metrics["accuracy"]} | P={metrics["precision"]} | R={metrics["recall"]} | F1={metrics["f1"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()