## Automated-Polyp-Segmentation

The lack of big, diversified, and well-annotated datasets makes it difficult to construct reliable automated segmentation models for colonoscopy images. To overcome this difficulty, the work creates a synthetic colonoscopy dataset by mask generation with the Segment Anything Model (SAM) and image synthesis using optimized Stable Diffusion. 
This repository contains an implementation of multiple deep learning architectures for medical image segmentation, specifically focused on polyp detection in colonoscopy images. The project includes training pipelines, model evaluation, and inference code for five different segmentation models.



![Model Architecture](./images/architecture.png)


## Features

- Implementation of 5 state-of-the-art segmentation architectures:
  - U-Net
  - PSPNet
  - FPN (Feature Pyramid Network)
  - LinkNet
  - MANet
- Comprehensive training pipeline with data augmentation
- Mixed precision training for improved performance
- Advanced loss function combining BCE, Dice, and Focal losses
- Model evaluation with multiple metrics
- Inference pipeline with visualization capabilities

## Project Structure

```
SPIE_Paper/
├── synthetic/           # Directory containing input images
├── masks/              # Directory containing ground truth masks
├── model_weights/      # Saved model weights
├── training_plots/     # Training visualizations and metrics
├── inference_results/  # Model inference results
└── results/           # Performance metrics and comparisons
```

## Requirements

```
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
numpy
matplotlib
tqdm
efficientnet-pytorch
torch-optimizer
```

Install requirements using:
```bash
pip install -r requirements.txt
```

## Model Architectures

The project implements multiple segmentation architectures as shown in the diagram above. All models use a ResNet-34 backbone pretrained on ImageNet. The architectures implemented are:

1. **U-Net**: A classic encoder-decoder architecture with skip connections
2. **PSPNet**: Pyramid Scene Parsing Network with hierarchical feature fusion
3. **FPN**: Feature Pyramid Network with lateral connections
4. **LinkNet**: Lightweight architecture with efficient skip connections
5. **MANet**: Multi-scale Attention Network

## Training Pipeline

![Training Process](./images/training_flow.png)

The training pipeline follows the process shown above and includes:

1. **Data Preparation**
   ```python
   dataset = PolypsDataset(img_dir, mask_dir, transform=train_transform)
   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   ```

2. **Data Augmentation**
   - Random rotations
   - Horizontal/vertical flips
   - Elastic transformations
   - Color augmentations
   - Normalization

3. **Loss Function**
   ```python
   class SegmentationLoss(nn.Module):
       def __init__(self, weights=[0.3, 0.4, 0.3]):
           self.bce = nn.BCEWithLogitsLoss()
           self.dice = smp.losses.DiceLoss(mode='binary')
           self.focal = smp.losses.FocalLoss(mode='binary')
   ```

4. **Training Loop**
   - Mixed precision training
   - Learning rate scheduling
   - Model checkpointing
   - Metric tracking

## Evaluation Metrics

The models are evaluated using multiple metrics:

- IoU (Intersection over Union)
- Dice coefficient
- Precision
- Recall
- F1 Score
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

## Inference

The inference pipeline provides:

1. Batch processing of images
2. Visualization of results
3. Comparison between different models
4. Overlay generation of predictions

Example usage:
```python
python run_inference_pipeline.py
```

## Results Visualization

![Results Comparison](./images/results_comparison.png)

Results are saved in multiple formats:

1. Individual model predictions
2. Side-by-side comparisons
3. Overlay visualizations
4. Metric comparisons

## Code Examples

### Loading a Model
```python
def get_model(num_classes=2):
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model
```

### Running Inference
```python
def run_inference(model, image, device):
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        pred_mask = torch.sigmoid(output).cpu().numpy().squeeze()
        return (pred_mask > 0.5).astype(np.uint8)
```

## Performance Tips

1. Use mixed precision training for faster computation
2. Enable data prefetching with `num_workers` in DataLoader
3. Use appropriate batch sizes based on available GPU memory
4. Apply gradient clipping to prevent exploding gradients

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Segmentation Models PyTorch library
- Albumentations library for augmentations
- PyTorch community for resources and inspiration

## Contact

For any queries or suggestions, please open an issue in the repository.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{medical_image_segmentation,
  author = {Ejiga Peter Ojonugwa Oluwafemi},
  title = {Medical Image Segmentation with Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ejigsonpeter/Automated-Polyp-Segmentation}
}
```
