# Retail Image Classifier

Deep Learning CNN para clasificación de ropa retail. Entrenado con **DeepFashion InShop** (17K fotos reales de estudio), MobileNetV3-Small, 78.5% accuracy en 12 categorías de ropa.

## Demo

**Live:** https://retail-image-classifier.vercel.app

Pipeline de preprocesamiento visible: resize → crop → normalize → inferencia CNN.

## Métricas

| Métrica | Valor |
|---------|-------|
| Accuracy | 78.5% |
| Precision | 78.6% |
| Recall | 78.7% |
| F1-Score | 78.5% |
| Parámetros | 1.5M |
| Epochs | 8 |

## Clases (12)

blouses, cardigans, dresses, graphic, jackets, pants, rompers, shorts, skirts, sweaters, sweatshirts, tees

## Stack

- **Training:** Python, PyTorch, ROCm (AMD RX 6600 XT)
- **Model:** MobileNetV3-Small (transfer learning via torchvision)
- **Dataset:** DeepFashion InShop (52K → 17.7K filtrado)
- **Export:** ONNX opset 18 (weights embedded, single file)
- **Demo:** ONNX Runtime Web 1.20, vanilla JS, preprocessing visualization
- **Hosting:** Vercel

## Archivos

```
├── demo/index.html         # Demo web interactivo
├── train_deepfashion2.py   # Training script principal
├── train.py                # Alternativo (Fashion-MNIST)
├── train_real.py           # Alternativo (Fashion Products)
├── requirements.txt
├── models/
│   ├── best_model_deepfashion.pth          # Checkpoint PyTorch
│   └── retail_classifier_deepfashion.onnx  # ONNX embedded
├── metrics/
│   ├── metrics_deepfashion.json
│   └── confusion_matrix_deepfashion.png
└── data/deepfashion/        # 12 subdirectories, 17.7K images
```

## Deployment

```bash
cd demo
vercel --prod
```

## Autor

David Daza — [GitHub](https://github.com/daviddaza2906)
