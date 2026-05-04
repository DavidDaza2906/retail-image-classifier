import os, json, random, time, copy, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.002
RANDOM_SEED = 42
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

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

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_preds = nn.functional.log_softmax(pred, dim=-1)
        nll = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        return (1 - self.smoothing) * nll.mean() + self.smoothing * smooth_loss.mean()

def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
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
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix — DeepFashion + Mixup + LabelSmoothing')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print('=' * 60)
    print('Retail Image Classifier — DeepFashion v2 (Improved)')
    print(f'Mixup α={MIXUP_ALPHA} | LabelSmoothing {LABEL_SMOOTHING} | OneCycleLR')
    print('=' * 60)

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load dataset from disk
    print(f'\nLoading DeepFashion images from disk...')
    full_ds = datasets.ImageFolder(DATA_DIR, transform=get_train_transform())
    class_names = full_ds.classes
    num_classes = len(class_names)
    print(f'Classes: {class_names}')
    print(f'Total images: {len(full_ds)}')

    labels = torch.tensor(full_ds.targets)
    for i, name in enumerate(class_names):
        print(f'  {name}: {(labels == i).sum().item()}')

    # Split
    total = len(full_ds)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    test_indices = test_ds.indices

    # Validation set uses val transform
    val_ds.dataset.transform = get_val_transform()

    # Test set loaded separately with val transform
    test_full = datasets.ImageFolder(DATA_DIR, transform=get_val_transform())
    test_subset = torch.utils.data.Subset(test_full, test_indices)

    print(f'\nTrain: {train_size}, Val: {val_size}, Test: {test_size}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    print(f'\nBuilding MobileNetV3-Small ({num_classes} classes)...')
    model = models.mobilenet_v3_small(weights='DEFAULT')
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = LabelSmoothingLoss(num_classes, smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.1, div_factor=10, final_div_factor=100
    )

    # Train
    start = time.time()
    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, 'best_model_v2.pth')

    for epoch in range(NUM_EPOCHS):
        e_start = time.time()
        use_mixup = epoch < NUM_EPOCHS - 2
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        e_time = time.time() - e_start
        lr = scheduler.get_last_lr()[0]
        suffix = f'  [Mixup]' if use_mixup else ''
        print(f'Epoch {epoch+1:2d}/{NUM_EPOCHS} | '
              f'Train: {train_loss:.4f}/{train_acc:.4f} | '
              f'Val: {val_loss:.4f}/{val_acc:.4f} | '
              f'{e_time:.0f}s | LR:{lr:.6f}{suffix}')
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  >>> Best val acc: {val_acc:.4f}')

    total_time = time.time() - start
    print(f'\nTraining done in {total_time:.0f}s ({total_time/60:.1f}min)')
    print(f'Best val acc: {best_val_acc:.4f}')

    # Evaluate
    print('\nEvaluating on test set...')
    model.load_state_dict(torch.load(best_path, weights_only=True))
    y_true, y_pred = predict(model, test_loader, device)
    metrics, cm = compute_metrics(y_true, y_pred, class_names)

    print(f'\nTest Accuracy:  {metrics["accuracy"]}')
    print(f'Test Precision: {metrics["precision"]}')
    print(f'Test Recall:    {metrics["recall"]}')
    print(f'Test F1:        {metrics["f1"]}')
    print('\nPer-class:')
    for cls, v in metrics['per_class'].items():
        print(f'  {cls:15s}: P={v["precision"]:.4f} R={v["recall"]:.4f} F1={v["f1"]:.4f}')

    with open(os.path.join(METRICS_DIR, 'metrics_v2.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    cm_path = os.path.join(METRICS_DIR, 'confusion_matrix_v2.png')
    plot_confusion_matrix(cm, cm_path, class_names)

    # Export ONNX
    print('\nExporting ONNX...')
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    onnx_tmp = os.path.join(MODELS_DIR, 'tmp_v2.onnx')
    torch.onnx.export(model, dummy, onnx_tmp,
                      input_names=['input'], output_names=['output'],
                      opset_version=18, do_constant_folding=True)

    from onnx import ModelProto, TensorProto, save as onnx_save
    import onnx
    m = onnx.load(onnx_tmp, load_external_data=False)
    with open(onnx_tmp + '.data', 'rb') as f:
        ext = f.read()
    for init in m.graph.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            off_val, len_val = None, None
            for ext_info in init.external_data:
                if ext_info.key == 'offset':
                    off_val = int(ext_info.value)
                elif ext_info.key == 'length':
                    len_val = int(ext_info.value)
            if off_val is not None:
                init.raw_data = ext[off_val:off_val + len_val]
                init.data_location = TensorProto.DEFAULT
                del init.external_data[:]

    final_onnx = os.path.join(MODELS_DIR, 'retail_classifier_final.onnx')
    onnx_save(m, final_onnx)

    import onnxruntime as ort
    s = ort.InferenceSession(final_onnx)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    r = s.run(None, {'input': x})
    print(f'ONNX: {os.path.getsize(final_onnx)/1024/1024:.2f}MB | Output: {r[0].shape}')

    print('\n' + '=' * 60)
    print(f'Model: {best_path}')
    print(f'ONNX:  {final_onnx}')
    print(f'Test Acc: {metrics["accuracy"]} | F1: {metrics["f1"]}')
    print('=' * 60)

if __name__ == '__main__':
    main()
