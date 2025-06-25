import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import random 
import glob
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
import albumentations as A
from tensorflow.keras.optimizers import Adam
import json
from datetime import datetime
from PIL import Image
import matplotlib.patches as patches
from collections import Counter

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Paths to image and mask folders
image_path = "./dubai_dataset/images/*.jpg"
mask_path = "./dubai_dataset/masks/*.png"

# Function to display images from a folder
def display_images(image_folder, num_images=9):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    
    for i, file_name in enumerate(image_files[:num_images]):
        image = cv2.imread(os.path.join(image_folder, file_name))
        axes[i // 3, i % 3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i // 3, i % 3].set_title(file_name)
        axes[i // 3, i % 3].axis('off')
        
    plt.tight_layout()
    plt.show()

# Display images and masks
display_images("./dubai_dataset/images")
display_images("./dubai_dataset/masks")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class DubaiAerialUNet:
    def __init__(self, input_size=(256, 256, 3), num_classes=6, classes_json_path=None):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        # Load class information from JSON file or use defaults
        if classes_json_path and os.path.exists(classes_json_path):
            self.class_names, self.class_colors = self.load_classes_from_json(classes_json_path)
            print(f"Loaded class information from {classes_json_path}")
            
            # Override with actual mask colors (based on diagnostic results)
            print("⚠️  Using actual mask colors instead of classes.json colors")
            self.class_colors = {
                0: [132, 41, 246],   # Land (most frequent) - was purple in old scheme
                1: [110, 193, 228],  # Road (blue) - was road in old scheme  
                2: [226, 169, 41],   # Water (orange) - was water in old scheme
                3: [155, 155, 155],  # Unlabeled (gray) - matches classes.json
                4: [254, 221, 58],   # Vegetation (yellow) - was vegetation in old scheme
                5: [60, 16, 152]     # Building (dark purple) - was building in old scheme
            }
            
            # Update class names to match the actual distribution
            self.class_names = [
                'Land (unpaved area)',  # Most frequent color (75.7%)
                'Road',                 # Second most frequent (14.2%) 
                'Water',                # Third most frequent (7.1%)
                'Unlabeled',            # Fourth most frequent (1.5%)
                'Vegetation',           # Fifth most frequent (0.9%)
                'Building'              # Least frequent (0.5%)
            ]
            
        else:
            # Fallback to actual mask colors found in diagnostic
            self.class_names = [
                'Land (unpaved area)',
                'Road', 
                'Water',
                'Unlabeled',
                'Vegetation',
                'Building'
            ]
            
            # RGB color codes based on actual mask analysis
            self.class_colors = {
                0: [132, 41, 246],   # Land (purple) - 75.7%
                1: [110, 193, 228],  # Road (blue) - 14.2%
                2: [226, 169, 41],   # Water (orange) - 7.1%
                3: [155, 155, 155],  # Unlabeled (gray) - 1.5%
                4: [254, 221, 58],   # Vegetation (yellow) - 0.9%
                5: [60, 16, 152]     # Building (dark purple) - 0.5%
            }
            print("Using actual mask colors from diagnostic analysis")
        
        # Create color map for visualization
        self.colormap = np.array([self.class_colors[i] for i in range(self.num_classes)], dtype=np.uint8)
        
        print("Dubai Aerial Imagery U-Net Initialized")
        print(f"Classes: {self.class_names}")
        print(f"Input size: {self.input_size}")
    
    def load_classes_from_json(self, json_path):
        """Load class information from the provided classes.json file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        class_names = []
        class_colors = {}
        
        for i, class_info in enumerate(data['classes']):
            class_names.append(class_info['title'])
            
            # Convert hex color to RGB
            hex_color = class_info['color'].lstrip('#')
            rgb_color = [int(hex_color[j:j+2], 16) for j in (0, 2, 4)]
            class_colors[i] = rgb_color
        
        return class_names, class_colors
    
    def rgb_to_class_mask(self, rgb_mask):
        """Convert RGB mask to class indices using actual mask colors"""
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use exact color matching with the actual colors found in masks
        for class_idx, target_color in self.class_colors.items():
            exact_mask = np.all(rgb_mask == target_color, axis=-1)
            class_mask[exact_mask] = class_idx
        
        return class_mask
    
    def class_mask_to_rgb(self, class_mask):
        """Convert class indices back to RGB visualization"""
        return self.colormap[class_mask]
    
    def load_dubai_dataset(self, images_dir, masks_dir, target_size=(256, 256)):
        """Load and preprocess Dubai aerial imagery dataset"""
        print("Loading Dubai Aerial Imagery dataset...")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        image_files.sort()
        print(f"Found {len(image_files)} images")
        
        # Get corresponding mask files
        mask_files = []
        for img_file in image_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            
            # Try different mask naming conventions
            possible_mask_names = [
                f"{base_name}.png",
                f"{base_name}_mask.png",
                f"{base_name}.jpg",
                f"{base_name}.tif"
            ]
            
            mask_found = False
            for mask_name in possible_mask_names:
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):
                    mask_files.append(mask_path)
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for {img_file}")
        
        print(f"Found {len(mask_files)} corresponding masks")
        
        if len(image_files) != len(mask_files):
            print("Warning: Mismatch between number of images and masks!")
            min_count = min(len(image_files), len(mask_files))
            image_files = image_files[:min_count]
            mask_files = mask_files[:min_count]
        
        # Load and process images
        images = []
        masks = []
        
        for img_path, mask_path in zip(image_files, mask_files):
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Could not load image: {img_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load mask
                mask = cv2.imread(mask_path)
                if mask is None:
                    print(f"Could not load mask: {mask_path}")
                    continue
                    
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                
                # Resize both to target size
                image = cv2.resize(image, target_size)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                
                # Convert RGB mask to class indices
                class_mask = self.rgb_to_class_mask(mask)
                
                # Normalize image
                image = image.astype(np.float32) / 255.0
                
                # Convert class mask to one-hot encoding
                mask_one_hot = tf.keras.utils.to_categorical(class_mask, num_classes=self.num_classes)
                
                images.append(image)
                masks.append(mask_one_hot)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(images)} image-mask pairs")
        
        if len(images) == 0:
            raise ValueError("No valid image-mask pairs found!")
        
        return np.array(images), np.array(masks)
    
    def get_augmentation_pipeline(self):
        """Augmentation pipeline optimized for aerial imagery"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=45, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.OneOf([
                A.ElasticTransform(p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=0.5),
            ], p=0.3)
        ])
    
    def augment_dataset(self, images, masks, augmentation_factor=8):
        """Augment the small Dubai dataset"""
        print(f"Augmenting dataset by factor of {augmentation_factor}...")
        
        augmentation = self.get_augmentation_pipeline()
        augmented_images = []
        augmented_masks = []
        
        # Keep original data
        augmented_images.extend(images)
        augmented_masks.extend(masks)
        
        # Generate augmented data
        for _ in range(augmentation_factor):
            for i, (image, mask) in enumerate(zip(images, masks)):
                # Convert one-hot mask back to class indices for augmentation
                class_mask = np.argmax(mask, axis=-1).astype(np.uint8)
                
                # Apply augmentation
                augmented = augmentation(image=image, mask=class_mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # Convert back to one-hot
                aug_mask_one_hot = tf.keras.utils.to_categorical(aug_mask, num_classes=self.num_classes)
                
                augmented_images.append(aug_image)
                augmented_masks.append(aug_mask_one_hot)
        
        print(f"Dataset augmented from {len(images)} to {len(augmented_images)} samples")
        return np.array(augmented_images), np.array(augmented_masks)
    
    def create_unet_model(self, dropout_rate=0.3, filters_base=32):
        """Create U-Net model optimized for Dubai aerial imagery"""
        inputs = layers.Input(self.input_size)
        
        # Encoder with batch normalization and dropout
        def conv_block(x, filters, dropout=False):
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            if dropout:
                x = layers.Dropout(dropout_rate)(x)
            return x
        
        # Encoder path
        enc1 = conv_block(inputs, filters_base)
        pool1 = layers.MaxPooling2D((2, 2))(enc1)
        
        enc2 = conv_block(pool1, filters_base * 2)
        pool2 = layers.MaxPooling2D((2, 2))(enc2)
        
        enc3 = conv_block(pool2, filters_base * 4)
        pool3 = layers.MaxPooling2D((2, 2))(enc3)
        
        enc4 = conv_block(pool3, filters_base * 8)
        pool4 = layers.MaxPooling2D((2, 2))(enc4)
        
        # Bottleneck
        bottleneck = conv_block(pool4, filters_base * 16, dropout=True)
        
        # Decoder path with skip connections
        def upconv_block(x, skip, filters, dropout=False):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Conv2D(filters, (2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Concatenate()([x, skip])
            x = conv_block(x, filters, dropout)
            return x
        
        # Decoder
        dec4 = upconv_block(bottleneck, enc4, filters_base * 8, dropout=True)
        dec3 = upconv_block(dec4, enc3, filters_base * 4)
        dec2 = upconv_block(dec3, enc2, filters_base * 2)
        dec1 = upconv_block(dec2, enc1, filters_base)
        
        # Output layer for multi-class segmentation
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(dec1)
        
        model = Model(inputs=inputs, outputs=outputs, name='Dubai_Aerial_UNet')
        return model
    
    def focal_loss(self, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance in aerial imagery"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Calculate cross entropy
            ce = -y_true * tf.math.log(y_pred)
            
            # Calculate focal weight
            weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
            
            # Apply focal weight
            focal_loss = weight * ce
            
            return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
        
        return focal_loss_fixed
    
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """Dice coefficient for multi-class segmentation"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def mean_iou_metric(self, y_true, y_pred):
        """Mean IoU for multi-class segmentation"""
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        iou_list = []
        for class_id in range(self.num_classes):
            true_class = tf.equal(y_true, class_id)
            pred_class = tf.equal(y_pred, class_id)
            
            intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_class, pred_class), tf.float32))
            union = tf.reduce_sum(tf.cast(tf.logical_or(true_class, pred_class), tf.float32))
            
            # Avoid division by zero
            iou = tf.cond(
                tf.equal(union, 0),
                lambda: 1.0,
                lambda: intersection / union
            )
            iou_list.append(iou)
        
        return tf.reduce_mean(iou_list)
    
    def build_model(self, learning_rate=1e-4, filters_base=32):
        """Build and compile the model"""
        self.model = self.create_unet_model(filters_base=filters_base)
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(alpha=0.25, gamma=2.0),
            metrics=['accuracy', self.dice_coefficient, self.mean_iou_metric]
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=8, 
                   model_save_path='best_dubai_unet_model.keras'):
        """Train the model with Dubai-specific settings"""
        
        callbacks = [
            ModelCheckpoint(
                model_save_path, 
                monitor='val_mean_iou_metric', 
                mode='max', 
                save_best_only=True, 
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=8, 
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def visualize_predictions(self, X_test, y_test, num_samples=6):
        """Visualize predictions with Dubai-specific colors"""
        if self.model is None:
            print("Model not built. Build the model first.")
            return
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        for i in range(num_samples):
            idx = random.randint(0, len(X_test) - 1)
            
            image = X_test[idx]
            true_mask = y_test[idx]
            
            # Predict
            pred_mask = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            
            # Convert to class indices
            true_classes = np.argmax(true_mask, axis=-1)
            pred_classes = np.argmax(pred_mask, axis=-1)
            
            # Convert to RGB for visualization
            true_rgb = self.class_mask_to_rgb(true_classes)
            pred_rgb = self.class_mask_to_rgb(pred_classes)
            
            # Calculate IoU
            iou = self.calculate_iou_multiclass(true_classes, pred_classes)
            
            # Display
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Aerial Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(true_rgb)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title(f'Prediction\nMean IoU: {iou:.3f}')
            axes[i, 2].axis('off')
            
            # Show class distribution
            class_counts = np.bincount(pred_classes.flatten(), minlength=self.num_classes)
            class_percentages = class_counts / np.sum(class_counts) * 100
            
            bars = axes[i, 3].bar(range(self.num_classes), class_percentages, 
                                color=[np.array(self.class_colors[j])/255.0 for j in range(self.num_classes)])
            axes[i, 3].set_title('Class Distribution')
            axes[i, 3].set_xlabel('Class')
            axes[i, 3].set_ylabel('Percentage')
            axes[i, 3].set_xticks(range(self.num_classes))
            axes[i, 3].set_xticklabels([name[:4] for name in self.class_names], rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_iou_multiclass(self, y_true, y_pred):
        """Calculate mean IoU for multi-class segmentation"""
        ious = []
        for class_id in range(self.num_classes):
            true_class = (y_true == class_id)
            pred_class = (y_pred == class_id)
            
            intersection = np.logical_and(true_class, pred_class).sum()
            union = np.logical_or(true_class, pred_class).sum()
            
            if union == 0:
                ious.append(1.0)  # Perfect score if no pixels of this class
            else:
                ious.append(intersection / union)
        
        return np.mean(ious)
    
    def create_class_legend(self):
        """Create a color legend for the classes"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, self.class_colors.values())):
            rect = patches.Rectangle((i, 0), 1, 1, 
                                   facecolor=np.array(color)/255.0, 
                                   edgecolor='black')
            ax.add_patch(rect)
            ax.text(i + 0.5, 0.5, class_name, 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, len(self.class_names))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Dubai Aerial Imagery - Class Color Legend', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_dataset(self, masks):
        """Analyze class distribution in the dataset"""
        print("Dataset Analysis:")
        print("=" * 50)
        
        total_pixels = 0
        class_pixels = np.zeros(self.num_classes)
        
        for mask in masks:
            class_indices = np.argmax(mask, axis=-1)
            unique, counts = np.unique(class_indices, return_counts=True)
            
            total_pixels += class_indices.size
            for class_id, count in zip(unique, counts):
                class_pixels[class_id] += count
        
        print(f"Total images: {len(masks)}")
        print(f"Total pixels: {total_pixels:,}")
        print("\nClass distribution:")
        
        for i, (class_name, pixel_count) in enumerate(zip(self.class_names, class_pixels)):
            percentage = (pixel_count / total_pixels) * 100
            print(f"{class_name:12}: {pixel_count:10.0f} pixels ({percentage:5.2f}%)")
        
        # Visualize distribution
        plt.figure(figsize=(10, 6))
        colors = [np.array(self.class_colors[i])/255.0 for i in range(self.num_classes)]
        bars = plt.bar(self.class_names, class_pixels, color=colors)
        plt.title('Class Distribution in Dubai Aerial Dataset')
        plt.xlabel('Class')
        plt.ylabel('Number of Pixels')
        plt.xticks(rotation=45)
        
        # Add percentage labels on bars
        for bar, pixels in zip(bars, class_pixels):
            height = bar.get_height()
            percentage = (pixels / total_pixels) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main training pipeline for Dubai Aerial Imagery"""
    print("Dubai Aerial Imagery U-Net Training Pipeline")
    print("=" * 60)
    
    # Configuration optimized for Dubai dataset
    CONFIG = {
        'images_dir': './dubai_dataset/images',
        'masks_dir': './dubai_dataset/masks',
        'classes_json': './dubai_dataset/classes.json',  # Path to classes.json
        'input_size': (256, 256, 3),
        'batch_size': 4,  # Small batch size due to limited data
        'epochs': 100,
        'learning_rate': 1e-4,
        'filters_base': 32,
        'augmentation_factor': 8,  # Heavy augmentation due to small dataset
        'test_size': 0.2,
        'validation_size': 0.25,
        'model_save_path': 'best_dubai_aerial_unet.keras'
    }
    
    # Initialize Dubai U-Net with classes.json
    dubai_unet = DubaiAerialUNet(
        input_size=CONFIG['input_size'],
        num_classes=6,
        classes_json_path=CONFIG['classes_json']
    )
    
    # Show class legend
    dubai_unet.create_class_legend()
    
    try:
        # Load dataset
        print("\n" + "="*50)
        print("LOADING DATASET")
        print("="*50)
        
        images, masks = dubai_unet.load_dubai_dataset(
            CONFIG['images_dir'], 
            CONFIG['masks_dir'],
            target_size=CONFIG['input_size'][:2]
        )
        
        # Analyze dataset
        dubai_unet.analyze_dataset(masks)
        
        # Augment dataset due to small size
        print("\n" + "="*50)
        print("AUGMENTING DATASET")
        print("="*50)
        
        aug_images, aug_masks = dubai_unet.augment_dataset(
            images, masks, 
            augmentation_factor=CONFIG['augmentation_factor']
        )
        
        # Split data
        print("\n" + "="*50)
        print("SPLITTING DATASET")
        print("="*50)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            aug_images, aug_masks, 
            test_size=CONFIG['test_size'], 
            random_state=42,
            stratify=None  # Can't stratify with multi-class masks easily
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=CONFIG['validation_size'], 
            random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples") 
        print(f"Test set: {len(X_test)} samples")
        
        # Build model
        print("\n" + "="*50)
        print("BUILDING MODEL")
        print("="*50)
        
        model = dubai_unet.build_model(
            learning_rate=CONFIG['learning_rate'],
            filters_base=CONFIG['filters_base']
        )
        
        # Train model
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        history = dubai_unet.train_model(
            X_train, y_train, X_val, y_val,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            model_save_path=CONFIG['model_save_path']
        )
        
        # Visualize results
        print("\n" + "="*50)
        print("VISUALIZING RESULTS")
        print("="*50)
        
        dubai_unet.visualize_predictions(X_test, y_test, num_samples=6)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best model saved as: {CONFIG['model_save_path']}")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        print("Please check your dataset paths and structure.")
        
        # Print expected structure
        print("\nExpected dataset structure:")
        print("dubai_dataset/")
        print("├── images/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg") 
        print("│   └── ...")
        print("└── masks/")
        print("    ├── image1.png")
        print("    ├── image2.png")
        print("    └── ...")

if __name__ == "__main__":
    main()