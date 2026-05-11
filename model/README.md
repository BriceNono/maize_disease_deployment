# Model Directory

Place your trained model file here:

```
model/
└── vgg16_maize_best.h5     ← copy from Google Colab SAVE_DIR
```

## How to export from Colab

After training, run Cell 13 (ModelExporter). Then download:
```
/content/drive/MyDrive/MaizeCNN_Results/vgg16_maize_best.h5
```
and place it in this folder.

## Class Order (must match class_indices from training)

Index 0 → Blight (Northern Leaf Blight)
Index 1 → Common_Rust
Index 2 → Gray_Leaf_Spot
Index 3 → Healthy

Verify with: `aug_pipeline.train_gen.class_indices` in Cell 5.
