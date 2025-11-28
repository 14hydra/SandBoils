# SandBoils
# Synthetic + Real Sand Boil Semantic Segmentation  
**High-Performance Detection of Sand Boils in Levee/Dam Imagery Using Deep Learning**

### Project Overview
This project trains a deep learning semantic segmentation model to automatically detect **sand boils** (also known as sand volcanoes or piping failures) in real-world and synthetic levee inspection images.  
Sand boils are critical indicators of internal erosion and potential levee failure — early detection can prevent catastrophic breaches.

**Best Result Achieved**: **0.65+ Dice Score** on real-world test images  
**Model Used**: `U-Net` with `swinv2_small_window16_256` or `efficientnet-b4` backbone  
**Framework**: PyTorch + segmentation-models-pytorch (SMP) + Albumentations

---

### Project Structure & Code Index

| Section | Purpose | Key Code / Notes |
|-------|--------|------------------|
| 1. Mount Google Drive | Access dataset and save models | `drive.mount('/content/drive')` |
| 2. Install Dependencies | Required libraries | `pip install torch timm segmentation-models-pytorch albumentations` |
| 3. Imports & Reproducibility | Set seed for full determinism | `seed = 3`, manual seeds for torch, numpy, etc. |
| 4. Custom Dataset Class | `SandBoilDataset` – loads paired image/mask from folders | Handles both `.jpg` + `.png` and `_mask.png` naming |
| 5. Data Augmentation | Albumentations transforms (strong augmentations for synthetic data) | `HorizontalFlip`, `RandomBrightnessContrast`, `Normalize` |
| 6. Dataset Paths & Splits | Training: Synthetic images<br>Validation/Test: Real images | Uses real images only for final evaluation |
| 7. DataLoader Setup | `train_loader`, `val_loader`, `test_loader` | Batch size = 8, proper shuffling |
| 8. Model Builder (`model_builder`) | Supports multiple architectures | **Recommended**: `"EfficientNet"` → EfficientNet B4 |
| 9. Loss Function | Combined loss | `CombinedLoss` |
| 10. Training Loop | Full training + validation with Dice tracking | Saves best model by **highest validation Dice** |
| 11. Best Model Save | `best_sandboil_model.pth` | Saved automatically when val Dice improves |
| 12. Visualization Functions | `show_prediction()`, `visualize_dataset()` | Includes proper denormalization for display |
| 13. Test Evaluation | Final Dice score on unseen real images | Critical metric: **Test Dice > 0.60** |

---

### Recommended Final Configuration (Highest Performance)

```python
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 25

# Best model (as of Nov 2025)
model, device = model_builder("EfficientNet")  # Uses EfficientNet B4

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
loss_fn = smp.losses.TverskyLoss(mode='binary', from_logits=True, alpha=0.7)
# or: loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
