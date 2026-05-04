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

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.002
RANDOM_SEED = 42
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'deepfashion6')
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
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

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
    running_loss, correct, total = 0.0, 0, 0
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
    print('Retail Classifier — DeepFashion 6 classes')
    print(f'Mixup α={MIXUP_ALPHA} | LabelSmoothing={LABEL_SMOOTHING} | OneCycleLR')
    print('=' * 60)

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    full_ds = datasets.ImageFolder(DATA_DIR, transform=get_train_transform())
    classes = full_ds.classes
    n_classes = len(classes)
    print(f'\nClasses ({n_classes}): {classes}')
    print(f'Total images: {len(full_ds)}')

    labels = torch.tensor(full_ds.targets)
    for i, c in enumerate(classes):
        print(f'  {c}: {(labels == i).sum().item()}')

    total = len(full_ds)
    train_sz = int(0.70 * total)
    val_sz = int(0.15 * total)
    test_sz = total - train_sz - val_sz

    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(RANDOM_SEED))

    val_ds.dataset.transform = get_val_transform()
    test_full = datasets.ImageFolder(DATA_DIR, transform=get_val_transform())
    test_subset = torch.utils.data.Subset(test_full, test_ds.indices)

    print(f'\nTrain: {train_sz}, Val: {val_sz}, Test: {test_sz}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = models.mobilenet_v3_small(weights='DEFAULT')
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    model = model.to(device)
    print(f'\nParameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = LabelSmoothingLoss(n_classes, smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.1, div_factor=10, final_div_factor=100)

    start = time.time()
    best_val_acc = 0.0
    best_path = os.path.join(MODELS_DIR, 'best_model_6c.pth')

    for epoch in range(NUM_EPOCHS):
        e_start = time.time()
        use_mixup = epoch < NUM_EPOCHS - 2
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        e_time = time.time() - e_start
        mx = '[Mix]' if use_mixup else ''

        print(f'{epoch+1:2d}/{NUM_EPOCHS} | T:{train_loss:.3f}/{train_acc:.4f} | V:{val_loss:.3f}/{val_acc:.4f} | {e_time:.0f}s {mx}')
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'  >> Best {val_acc:.4f}')

    print(f'\nDone {time.time()-start:.0f}s | Best val: {best_val_acc:.4f}')

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

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f'\nTest Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}')
    for i, c in enumerate(classes):
        print(f'  {c:12s}: P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}')

    metrics = {
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4),
        'per_class': {classes[i]: {'precision': round(per_prec[i], 4),
                                     'recall': round(per_rec[i], 4),
                                     'f1': round(per_f1[i], 4)} for i in range(n_classes)}
    }
    with open(os.path.join(METRICS_DIR, 'metrics_6c.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=classes, yticklabels=classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, 'confusion_matrix_6c.png'), dpi=150)
    plt.close()

    # Export ONNX
    print('\nExporting ONNX...')
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    onnx_tmp = os.path.join(MODELS_DIR, 'tmp_6c.onnx')
    torch.onnx.export(model, dummy, onnx_tmp, input_names=['input'], output_names=['output'],
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
                if ext_info.key == 'offset': off_val = int(ext_info.value)
                elif ext_info.key == 'length': len_val = int(ext_info.value)
            if off_val is not None:
                init.raw_data = ext[off_val:off_val + len_val]
                init.data_location = TensorProto.DEFAULT
                del init.external_data[:]

    final_onnx = os.path.join(MODELS_DIR, 'retail_classifier_6c.onnx')
    onnx_save(m, final_onnx)

    import onnxruntime as ort
    s = ort.InferenceSession(final_onnx)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    s.run(None, {'input': x})
    print(f'ONNX embedded: {os.path.getsize(final_onnx)/1024/1024:.2f}MB')

    print('\n' + '=' * 60)
    print(f'Test Acc: {acc:.4f} | F1: {f1:.4f}')
    print(f'ONNX: {final_onnx}')
    print('=' * 60)

if __name__ == '__main__':
    main()
