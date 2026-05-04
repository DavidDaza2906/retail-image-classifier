import os, json, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets

BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 0.001
RANDOM_SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'deepfashion')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    running_loss, correct, total = 0.0, 0, 0
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

def main():
    print('=' * 60)
    print('Retail Image Classifier — DeepFashion InShop')
    print('MobileNetV3-Small')
    print('=' * 60)

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load from image folder
    print(f'\nLoading images from {DATA_DIR}...')
    full_ds = datasets.ImageFolder(DATA_DIR, transform=get_transform(True))
    print(f'Classes: {full_ds.classes}')
    print(f'Total images: {len(full_ds)}')

    # Class distribution
    labels = torch.tensor(full_ds.targets)
    print('\nClass distribution:')
    for i, name in enumerate(full_ds.classes):
        print(f'  {name}: {(labels == i).sum().item()}')

    # Split: 70% train, 15% val, 15% test
    total = len(full_ds)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Apply correct transform for val/test
    val_ds.dataset.transform = get_transform(False)

    # For test, load separately
    test_ds_full = datasets.ImageFolder(DATA_DIR, transform=get_transform(False))

    print(f'\nTrain: {train_size}, Val: {val_size}, Test: {test_size}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    test_indices = test_ds.indices
    test_subset = torch.utils.data.Subset(test_ds_full, test_indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    num_classes = len(full_ds.classes)
    print(f'\nBuilding MobileNetV3-Small ({num_classes} classes)...')
    model = models.mobilenet_v3_small(weights='DEFAULT')
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_p:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Train
    start = time.time()
    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, 'best_model_deepfashion.pth')

    for epoch in range(NUM_EPOCHS):
        e_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        e_time = time.time() - e_start
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '
              f'Train: {train_loss:.4f}/{train_acc:.4f} | '
              f'Val: {val_loss:.4f}/{val_acc:.4f} | '
              f'{e_time:.0f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  >>> Best {val_acc:.4f}')

    total_time = time.time() - start
    print(f'\nDone in {total_time:.0f}s ({total_time/60:.1f}min) | Best val: {best_val_acc:.4f}')

    # Evaluate
    print('\nEvaluating...')
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    class_names = full_ds.classes

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f'\nTest Accuracy:  {acc:.4f}')
    print(f'Test Precision: {prec:.4f}')
    print(f'Test Recall:    {rec:.4f}')
    print(f'Test F1:        {f1:.4f}')

    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    print('\nPer-class:')
    for i in range(len(class_names)):
        print(f'  {class_names[i]:15s}: P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}')

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'per_class': {class_names[i]: {
            'precision': round(per_prec[i], 4),
            'recall': round(per_rec[i], 4),
            'f1': round(per_f1[i], 4)
        } for i in range(len(class_names))}
    }
    with open(os.path.join(METRICS_DIR, 'metrics_deepfashion.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    fig.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, 'confusion_matrix_deepfashion.png'), dpi=150)
    plt.close()

    # Export ONNX
    print('\nExporting ONNX...')
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = os.path.join(MODELS_DIR, 'temp_deepfashion.onnx')
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=18, do_constant_folding=True)

    # Embed weights
    print('Embedding weights...')
    from onnx import ModelProto, TensorProto, save as onnx_save
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    with open(onnx_path + '.data', 'rb') as f:
        ext = f.read()
    off = 0
    smap = {1: 4, 7: 4, 8: 8, 11: 8}
    for init in m.graph.initializer:
        ne = 1
        for d in init.dims:
            ne *= d
        es = smap.get(init.data_type, 4)
        init.raw_data = ext[off:off + ne * es]
        init.data_location = TensorProto.DEFAULT
        off += ne * es

    final_path = os.path.join(MODELS_DIR, 'retail_classifier_deepfashion.onnx')
    onnx_save(m, final_path)

    import onnxruntime as ort
    ort.InferenceSession(final_path)
    print(f'ONNX: {os.path.getsize(final_path) / 1024 / 1024:.2f} MB')

    print('\n=== Done ===')
    print(f'Model: {best_path}')
    print(f'ONNX:  {final_path}')
    print(f'Acc:   {acc:.4f} | F1: {f1:.4f}')

if __name__ == '__main__':
    main()
