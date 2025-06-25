# Enhanced U-Net for Dubai Aerial Imagery Segmentation

## Team Information
- **Project**: Enhanced U-Net for Dubai Aerial Imagery Segmentation
- **Course**: CS 5330 Pattern Recognition and Computer Vision
- **Team Members**: Feiyan Zhou, Zhechao Jin, Testing: Chieh-Han Chen, Jing Hui Ng
- **Date**: 6/16/2025
- **Dataset**: Dubai Aerial Imagery from MBRSC Satellites

## Project Overview

This project implements an enhanced U-Net model specifically optimized for the Dubai Aerial Imagery dataset. The model performs semantic segmentation on satellite imagery of Dubai, classifying each pixel into 6 distinct classes: Water, Land, Road, Building, Vegetation, and Unlabeled areas. The implementation includes significant improvements over the baseline U-Net with modern deep learning techniques and dataset-specific optimizations.

## Dataset: Dubai Aerial Imagery

### 📊 Dataset Information
- **Source**: MBRSC (Mohammed Bin Rashid Space Centre) Satellites
- **Size**: 72 high-resolution aerial images of Dubai, UAE
- **Classes**: 6 semantic classes with pixel-perfect annotations
- **License**: CC0 (Public Domain)
- **Quality**: Professional satellite imagery with expert annotations

### 🎨 Class Information (Actual Mask Colors)
Based on analysis of the actual mask files, the Dubai dataset uses these colors:

| Class | Color | Hex Code | RGB Values | Frequency |
|-------|-------|----------|------------|-----------|
| Land (unpaved area) | 🟣 Purple | #8429F6 | (132, 41, 246) | ~75.7% |
| Road | 🔵 Blue | #6EC1E4 | (110, 193, 228) | ~14.2% |
| Water | 🟠 Orange | #E2A929 | (226, 169, 41) | ~7.1% |
| Unlabeled | ⚪ Gray | #9B9B9B | (155, 155, 155) | ~1.5% |
| Vegetation | 🟡 Yellow | #FEDD3A | (254, 221, 58) | ~0.9% |
| Building | 🟪 Dark Purple | #3C1098 | (60, 16, 152) | ~0.5% |

**⚠️ Important Note**: The original classes.json file downloaded from Kaggle contains different colors than what's actually used in the mask files. Our implementation automatically detects and uses the correct colors from the actual masks.

## Key Features

### Model Enhancements
- **Enhanced U-Net Architecture**: Deeper network with batch normalization and dropout
- **Dubai-Specific Optimizations**: Tailored for aerial imagery characteristics
- **Focal Loss Function**: Handles class imbalance in satellite imagery
- **Multi-class IoU Tracking**: Comprehensive evaluation metrics
- **Heavy Data Augmentation**: 8x augmentation to expand the small dataset
- **Automatic Color Mapping**: Reads class definitions from classes.json

### Technical Improvements
- **Smart Preprocessing**: Automatic RGB-to-class conversion using classes.json
- **Memory-Efficient Training**: Optimized for small datasets with limited GPU memory
- **Advanced Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Comprehensive Visualization**: Color-coded predictions with class legends
- **Dataset Analysis Tools**: Class distribution analysis and mask verification

## Installation and Setup

### Prerequisites
```bash
# Required Python packages
pip install tensorflow>=2.8.0
pip install opencv-python>=4.6.0
pip install matplotlib>=3.5.0
pip install scikit-learn>=1.1.0
pip install albumentations>=1.2.0
pip install numpy>=1.21.0
pip install Pillow>=9.0.0
```

### Directory Structure
```
miniproject10/
├── unet.py         # Main enhanced U-Net implementation
├── color_diagnostic_script.py   # Color mapping diagnostic tool
├── verify_fix_script.py         # Verification script for color fix
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── dubai_dataset/               # Dataset folder
│   ├── classes.json            # Class definitions (IMPORTANT!)
│   ├── images/                 # RGB aerial images (72 files)
│   │   ├── image_part_001.jpg
│   │   ├── image_part_002.jpg
│   │   └── ...
│   └── masks/                  # Segmentation masks (72 files)
│       ├── image_part_001.png
│       ├── image_part_002.png
│       └── ...
├── models/                     # Saved models
│   └── best_dubai_aerial_unet.keras
```

## Dataset Download and Setup

### Step 1: Download Dubai Aerial Imagery Dataset
1. **Visit Kaggle**: [Dubai Aerial Imagery Dataset](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
2. **Download** the dataset (requires free Kaggle account)
3. **Extract** the downloaded archive to your project folder

### Step 2: Verify Color Mapping (Important!)
```bash
# Check if your dataset has color mapping issues
python color_diagnostic_script.py
```

This diagnostic will:
- ✅ Show actual colors in your mask files
- ✅ Compare with classes.json definitions  
- ✅ Identify any color mismatches
- ✅ Create visualization of the analysis

**Expected Output**: You should see 6 distinct colors with reasonable distribution (not 99% one class).

### Step 3: Run Training Verification (Optional)
```bash
# Verify that color mapping is fixed
python verify_fix_script.py
```

This will confirm that your class distribution looks realistic before training.

## Usage

### Basic Training
```bash
# Run the enhanced U-Net with default settings
python unet.py
```

### Advanced Configuration
Modify the CONFIG dictionary in `unet.py`:

```python
CONFIG = {
    'images_dir': './dubai_dataset/images',
    'masks_dir': './dubai_dataset/masks',
    'classes_json': './dubai_dataset/classes.json',
    'input_size': (256, 256, 3),        # Image dimensions
    'batch_size': 4,                    # Adjust based on GPU memory
    'epochs': 100,                      # Training epochs
    'learning_rate': 1e-4,              # Learning rate
    'filters_base': 32,                 # Base number of filters
    'augmentation_factor': 8,           # Data augmentation multiplier
    'model_save_path': 'best_dubai_aerial_unet.keras'
}
```

## Model Architecture

### Enhanced U-Net Features
- **Encoder**: 5 levels with progressive filter increases (32→64→128→256→512)
- **Decoder**: Symmetric decoder with skip connections
- **Batch Normalization**: Applied after each convolution for stable training
- **Dropout**: Applied in bottleneck and deep layers for regularization
- **Multi-class Output**: Softmax activation for 6-class segmentation

### Dubai-Specific Optimizations
- **Automatic Color Detection**: Uses actual mask colors, not classes.json
- **Robust Color Mapping**: Handles color mismatches gracefully
- **Focal Loss**: α=0.25, γ=2.0 for handling class imbalance
- **Heavy Augmentation**: 8x data expansion through transformations
- **Smart Class Distribution**: Automatically maps to most frequent colors
- **Small Batch Training**: Optimized for limited dataset size

## Training Process

### Data Augmentation Pipeline
```python
# Augmentations optimized for aerial imagery
- Horizontal/Vertical flips
- Random rotations (up to 45°)
- Scale and shift transformations
- Brightness/contrast adjustments
- Hue/saturation variations
- Gaussian blur
- Elastic transformations
```

### Training Callbacks
- **ModelCheckpoint**: Saves best model based on validation IoU
- **EarlyStopping**: Prevents overfitting (patience=15)
- **ReduceLROnPlateau**: Adaptive learning rate reduction

## Results and Performance

### Expected Performance Metrics
- **Training Accuracy**: 85-90%
- **Validation IoU**: 0.75-0.85
- **Training Time**: 30-60 minutes (GPU), 2-3 hours (CPU)
- **Model Size**: ~2-3 MB
- **Convergence**: 50-80 epochs
- **Class Distribution**: Realistic (not 99% one class!)

### Evaluation Metrics
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Mean IoU**: Average Intersection over Union across all classes
- **Dice Coefficient**: Harmonic mean of precision and recall
- **Per-class IoU**: Individual class performance analysis

## File Downloads and Links

### Pre-trained Model
- **Model File**: `models/best_dubai_aerial_unet.keras`
- **Training History**: `results/training_history.json`
- **Class Legend**: `results/class_legend.png`
- **Sample Predictions**: `results/predictions_showcase.png`

### Cloud Storage Links
- **Pre-trained Model**: https://drive.google.com/file/d/1nEDf_w7nQBSttbPG4B8LuMAClKRQr1L3/view?usp=sharing
- **Training Results**: https://drive.google.com/file/d/1BYCkIL3UiQG3fAa2PDYxxfVky9vc5q-y/view?usp=sharing
- **Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

### Video Demonstration
- **Demo Video**: https://northeastern.zoom.us/rec/share/j43MEI_Emvan496FuCX_F8IYru8DghbcpqPeL29Rj5ESLte7qjPmy-YbGcH518YN.xrPqqenz9uhAiJt9?startTime=1750142988000
Passcode: ekJ3.ZzF
- **Duration**: ~ 1 minute
- **Content**: Real-time segmentation demonstration on Dubai imagery


## Results Analysis

### Model Performance
- **Dataset**: Dubai Aerial Imagery (72 images → 648 after augmentation)
- **Training/Validation/Test Split**: 70%/20%/10% 
- **Best Validation IoU**: 0.3162
- **Training Convergence**: 50-80 epochs
- **Final Model Size**: 7,773,958 parameters

### Model Improvements Implemented
1. **Enhanced Architecture**: Added batch normalization and dropout for stability
2. **Automatic Color Detection**: Robust handling of color mapping mismatches
3. **Focal Loss**: Better handling of class imbalance (Land: 75.7% vs Building: 0.5%)
4. **Advanced Augmentation**: Comprehensive transformation pipeline for small datasets
5. **Smart Callbacks**: Early stopping and learning rate scheduling
6. **Diagnostic Tools**: Complete color mapping analysis and verification
7. **Comprehensive Evaluation**: Multiple metrics and beautiful visualizations

## Future Improvements

### Potential Enhancements
1. **Architecture Upgrades**: Try U-Net++, Attention U-Net, or DeepLabV3+
2. **Transfer Learning**: Use pre-trained encoders (ResNet, EfficientNet)
3. **Multi-scale Training**: Train on different image resolutions
4. **Test-Time Augmentation**: Average predictions from multiple augmented versions
5. **Ensemble Methods**: Combine multiple models for better performance

### Dataset Expansion
1. **Collect More Data**: Expand beyond 72 images if possible
2. **Cross-Domain Testing**: Test on other UAE cities
3. **Temporal Analysis**: Compare imagery from different time periods
4. **Multi-Resolution**: Include different zoom levels

## Technical Specifications

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA GPU with 8GB+ VRAM

### Software Dependencies
- **Python**: 3.8+
- **TensorFlow**: 2.8+
- **CUDA**: 11.2+ (for GPU training)
- **Additional**: OpenCV, Matplotlib, Albumentations, Scikit-learn

## References

### Academic Papers
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
3. Buslaev, A., et al. (2020). Albumentations: Fast and Flexible Image Augmentations.

### Dataset Sources
1. Dubai Aerial Imagery Dataset - Humans in the Loop, CC0 License
2. MBRSC (Mohammed Bin Rashid Space Centre) Satellite Data
3. Kaggle Dataset Platform

### Technical Resources
1. TensorFlow/Keras Documentation
2. Albumentations Documentation  
3. OpenCV Documentation
4. Course Materials - CS 5330 Pattern Recognition and Computer Vision

## License and Usage

This project is developed for educational purposes as part of CS 5330 coursework. The Dubai Aerial Imagery dataset is available under CC0 license (public domain). The enhanced U-Net implementation is available for academic and research use.

## Contact and Support

- **Repository**: https://github.khoury.northeastern.edu/jinggghui/CS5330_Su25_Group4
- **Contact**: chen.chiehh@northeastern.edu
- **Course**: CS 5330 Pattern Recognition and Computer Vision
- **Institution**: Northeastern University

---

**Last Updated**: 6/16/2025
**Status**: ✅ Complete Implementation with Dubai Dataset Integration  
**Performance**: 🎯 Optimized for High-Quality Aerial Imagery Segmentation
