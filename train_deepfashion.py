import os, json, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from datasets import load_dataset

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MAX_SAMPLES_PER_CLASS = 2000  # Limit to avoid OOM
DEEPFASHION_CLASSES = [
    'tees', 'blouses', 'dresses', 'shorts', 'sweaters',
    'pants', 'jackets', 'skirts', 'rompers', 'sweatshirts',
    'cardigans', 'graphic'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = img.convert('RGB') if img.mode != 'RGB' else img
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


def build_model(num_classes):
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


def compute_metrics(y_true, y_pred, class_names):
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
        'per_class': {class_names[i]: {
            'precision': round(per_prec[i], 4),
            'recall': round(per_rec[i], 4),
            'f1': round(per_f1[i], 4)
        } for i in range(len(class_names))}
    }, cm


def plot_confusion_matrix(cm, save_path, class_names):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix — DeepFashion (MobileNetV3)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def embed_and_verify_onnx(src_onnx, dst_onnx):
    from onnx import ModelProto, TensorProto, save as onnx_save
    import onnx

    model = onnx.load(src_onnx, load_external_data=False)

    with open(src_onnx + '.data', 'rb') as f:
        ext_data = f.read()

    offset = 0
    for init in model.graph.initializer:
        num_elems = 1
        for d in init.dims:
            num_elems *= d
        size_map = {1: 4, 7: 4, 8: 8, 11: 8}
        elem_size = size_map.get(init.data_type, 4)
        expected = num_elems * elem_size
        init.raw_data = ext_data[offset:offset + expected]
        init.data_location = TensorProto.DEFAULT
        offset += expected

    onnx_save(model, dst_onnx)

    import onnxruntime as ort
    sess = ort.InferenceSession(dst_onnx)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    r = sess.run(None, {'input': x})
    print(f'ONNX verified! Shape: {r[0].shape}')
    return dst_onnx


def main():
    print('=' * 60)
    print('Retail Image Classifier — DeepFashion InShop')
    print(f'MobileNetV3-Small + {len(DEEPFASHION_CLASSES)} classes')
    print('=' * 60)

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    start_time = time.time()

    # Load and filter dataset
    print(f'\n[1/7] Loading DeepFashion InShop dataset...')
    ds = load_dataset("Marqo/deepfashion-inshop", split="data", streaming=True)

    class_to_idx = {c: i for i, c in enumerate(DEEPFASHION_CLASSES)}
    all_samples = []
    class_counts = {c: 0 for c in DEEPFASHION_CLASSES}

    for sample in ds:
        cat = sample['category2']
        if cat in class_to_idx and class_counts[cat] < MAX_SAMPLES_PER_CLASS:
            all_samples.append((sample['image'], class_to_idx[cat]))
            class_counts[cat] += 1

    print(f'Filtered {len(all_samples)} images in {len(DEEPFASHION_CLASSES)} classes')

    # Show class distribution
    for c, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f'  {c}: {n}')

    # Stratified split
    random.shuffle(all_samples)
    test_size = int(len(all_samples) * TEST_SPLIT)
    val_size = int(len(all_samples) * VAL_SPLIT)
    train_size = len(all_samples) - test_size - val_size

    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]

    train_ds = DeepFashionDataset(train_samples, class_to_idx, get_transform(True))
    val_ds = DeepFashionDataset(val_samples, class_to_idx, get_transform(False))
    test_ds = DeepFashionDataset(test_samples, class_to_idx, get_transform(False))

    print(f'\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Build model
    print('\n[2/7] Building MobileNetV3-Small...')
    model = build_model(len(DEEPFASHION_CLASSES)).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {trainable:,}/{total:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Train
    print(f'\n[3/7] Training {NUM_EPOCHS} epochs...')
    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, 'best_model_deepfashion.pth')

    for epoch in range(NUM_EPOCHS):
        e_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        e_time = time.time() - e_start
        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '
              f'Train: {train_loss:.4f}/{train_acc:.4f} | '
              f'Val: {val_loss:.4f}/{val_acc:.4f} | '
              f'{e_time:.0f}s | LR:{lr:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  >>> Best model {val_acc:.4f}')

    total_time = time.time() - start_time
    print(f'\nTraining done in {total_time:.0f}s ({total_time/60:.1f}min) | Best val: {best_val_acc:.4f}')

    # Evaluate
    print('\n[4/7] Evaluating...')
    model.load_state_dict(torch.load(best_path, weights_only=True))
    y_true, y_pred = predict(model, test_loader, device)
    metrics, cm = compute_metrics(y_true, y_pred, DEEPFASHION_CLASSES)

    print(f'\nTest Accuracy:  {metrics["accuracy"]}')
    print(f'Test Precision: {metrics["precision"]}')
    print(f'Test Recall:    {metrics["recall"]}')
    print(f'Test F1:        {metrics["f1"]}')
    print('\nPer-class:')
    for cls, v in metrics['per_class'].items():
        print(f'  {cls:15s}: P={v["precision"]:.4f} R={v["recall"]:.4f} F1={v["f1"]:.4f}')

    with open(os.path.join(METRICS_DIR, 'metrics_deepfashion.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(cm, os.path.join(METRICS_DIR, 'confusion_matrix_deepfashion.png'), DEEPFASHION_CLASSES)

    # Export ONNX
    print('\n[5/7] Exporting ONNX...')
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    export_path = os.path.join(MODELS_DIR, 'deepfashion_temp.onnx')
    torch.onnx.export(model, dummy, export_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=18, do_constant_folding=True)

    # Embed weights into single file
    print('\n[6/7] Embedding weights...')
    final_path = os.path.join(MODELS_DIR, 'retail_classifier_deepfashion.onnx')
    embed_and_verify_onnx(export_path, final_path)

    size = os.path.getsize(final_path) / 1024 / 1024
    print(f'Final ONNX: {size:.2f} MB')

    print('\n[7/7] Summary')
    print('=' * 60)
    print(f'  Dataset:  DeepFashion InShop ({len(all_samples)} imgs)')
    print(f'  Classes:  {len(DEEPFASHION_CLASSES)}: {", ".join(DEEPFASHION_CLASSES[:6])}...')
    print(f'  Model:    MobileNetV3-Small')
    print(f'  Epochs:   {NUM_EPOCHS}')
    print(f'  Test Acc: {metrics["accuracy"]} | P={metrics["precision"]} | R={metrics["recall"]} | F1={metrics["f1"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()