import cv2
import numpy as np
import matplotlib.pyplot as plt

def verify_color_fix():
    """Verify that the color mapping is now working correctly"""
    
    # Actual colors found in your masks (from diagnostic)
    actual_colors = {
        0: [132, 41, 246],   # Land (purple) - 75.7%
        1: [110, 193, 228],  # Road (blue) - 14.2%
        2: [226, 169, 41],   # Water (orange) - 7.1%
        3: [155, 155, 155],  # Unlabeled (gray) - 1.5%
        4: [254, 221, 58],   # Vegetation (yellow) - 0.9%
        5: [60, 16, 152]     # Building (dark purple) - 0.5%
    }
    
    class_names = [
        'Land (unpaved area)',
        'Road', 
        'Water',
        'Unlabeled',
        'Vegetation',
        'Building'
    ]
    
    print("🔧 FIXED COLOR MAPPING")
    print("=" * 50)
    
    # Load a test mask
    mask_path = './dubai_dataset/masks/image_part_001.png'
    
    try:
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"❌ Could not load mask: {mask_path}")
            return
        
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Test the conversion
        h, w = mask_rgb.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Apply the fixed color mapping
        for class_idx, target_color in actual_colors.items():
            exact_mask = np.all(mask_rgb == target_color, axis=-1)
            class_mask[exact_mask] = class_idx
        
        # Calculate class distribution
        unique_classes, counts = np.unique(class_mask, return_counts=True)
        total_pixels = h * w
        
        print("✅ NEW CLASS DISTRIBUTION:")
        print("-" * 30)
        
        for class_idx, count in zip(unique_classes, counts):
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                percentage = (count / total_pixels) * 100
                print(f"{class_name:20}: {count:7d} pixels ({percentage:5.1f}%)")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original mask
        axes[0].imshow(mask_rgb)
        axes[0].set_title('Original RGB Mask')
        axes[0].axis('off')
        
        # Class mask (as image)
        # Create a colorful visualization of the class mask
        class_visualization = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in actual_colors.items():
            class_pixels = (class_mask == class_idx)
            class_visualization[class_pixels] = color
        
        axes[1].imshow(class_visualization)
        axes[1].set_title('Class Mask (Reconstructed)')
        axes[1].axis('off')
        
        # Class distribution bar chart
        valid_classes = [i for i in unique_classes if i < len(class_names)]
        valid_counts = [counts[list(unique_classes).index(i)] for i in valid_classes]
        valid_names = [class_names[i] for i in valid_classes]
        colors = [np.array(actual_colors[i])/255.0 for i in valid_classes]
        
        bars = axes[2].bar(range(len(valid_names)), valid_counts, color=colors)
        axes[2].set_title('Class Distribution')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('Pixel Count')
        axes[2].set_xticks(range(len(valid_names)))
        axes[2].set_xticklabels([name[:8] for name in valid_names], rotation=45)
        
        # Add percentage labels
        for bar, count in zip(bars, valid_counts):
            height = bar.get_height()
            percentage = (count / total_pixels) * 100
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('Dubai Dataset - Fixed Color Mapping', fontsize=16)
        plt.tight_layout()
        plt.savefig('fixed_color_mapping.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Color mapping verification complete!")
        print("Visualization saved as 'fixed_color_mapping.png'")
        
        # Verify no "all water" issue
        if len(unique_classes) > 1:
            print("✅ Multiple classes detected - 'all water' issue FIXED!")
        else:
            print("❌ Still showing only one class - needs further investigation")
            
    except Exception as e:
        print(f"Error during verification: {e}")

def create_corrected_classes_json():
    """Create a corrected classes.json file with actual colors"""
    
    corrected_classes = {
        "classes": [
            {"title": "Land (unpaved area)", "color": "#8429F6", "shape": "polygon", "geometry_config": {}},
            {"title": "Road", "color": "#6EC1E4", "shape": "polygon", "geometry_config": {}},
            {"title": "Water", "color": "#E2A929", "shape": "polygon", "geometry_config": {}},
            {"title": "Unlabeled", "color": "#9B9B9B", "shape": "polygon", "geometry_config": {}},
            {"title": "Vegetation", "color": "#FEDD3A", "shape": "polygon", "geometry_config": {}},
            {"title": "Building", "color": "#3C1098", "shape": "polygon", "geometry_config": {}}
        ],
        "tags": []
    }
    
    import json
    with open('corrected_classes.json', 'w') as f:
        json.dump(corrected_classes, f, indent=2)
    
    print("📝 Created 'corrected_classes.json' with actual mask colors")
    print("You can replace your original classes.json with this file")

if __name__ == "__main__":
    print("🔍 Verifying Color Mapping Fix...")
    verify_color_fix()
    print("\n" + "="*50)
    create_corrected_classes_json()
