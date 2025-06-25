import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import Counter

def diagnose_color_issues(mask_path, classes_json_path='./dubai_dataset/classes.json'):
    """Comprehensive color diagnosis for Dubai dataset masks"""
    
    print("🔍 DUBAI DATASET COLOR DIAGNOSTIC")
    print("=" * 50)
    
    # Load classes.json
    try:
        with open(classes_json_path, 'r') as f:
            classes_data = json.load(f)
        
        expected_colors = {}
        class_names = {}
        
        for i, class_info in enumerate(classes_data['classes']):
            class_names[i] = class_info['title']
            hex_color = class_info['color'].lstrip('#')
            rgb_color = [int(hex_color[j:j+2], 16) for j in (0, 2, 4)]
            expected_colors[i] = rgb_color
        
        print("Expected colors from classes.json:")
        for i, (name, color) in enumerate(zip(class_names.values(), expected_colors.values())):
            print(f"  {i}: {name:20} | RGB{tuple(color)}")
            
    except Exception as e:
        print(f"Error loading classes.json: {e}")
        return
    
    # Load and analyze mask
    try:
        print(f"\nAnalyzing mask: {mask_path}")
        
        # Try different loading methods
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            print("❌ Could not load mask file!")
            return
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"Mask shape: {mask_rgb.shape}")
        print(f"Mask dtype: {mask_rgb.dtype}")
        
        # Get unique colors
        unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
        print(f"\nFound {len(unique_colors)} unique colors in mask:")
        
        color_counts = []
        for color in unique_colors:
            count = np.sum(np.all(mask_rgb == color, axis=-1))
            color_counts.append((tuple(color), count))
        
        # Sort by frequency
        color_counts.sort(key=lambda x: x[1], reverse=True)
        
        print("\nColors by frequency:")
        for i, (color, count) in enumerate(color_counts):
            percentage = (count / (mask_rgb.shape[0] * mask_rgb.shape[1])) * 100
            print(f"  {i+1}: RGB{color} | {count:7d} pixels ({percentage:5.1f}%)")
        
        # Compare with expected colors
        print("\n🔍 COLOR MATCHING ANALYSIS:")
        print("-" * 50)
        
        for class_idx, (class_name, expected_color) in enumerate(zip(class_names.values(), expected_colors.values())):
            expected_tuple = tuple(expected_color)
            
            if expected_tuple in [c[0] for c in color_counts]:
                count = next(c[1] for c in color_counts if c[0] == expected_tuple)
                percentage = (count / (mask_rgb.shape[0] * mask_rgb.shape[1])) * 100
                print(f"✅ {class_name:20} | RGB{expected_tuple} | {count:7d} pixels ({percentage:5.1f}%)")
            else:
                print(f"❌ {class_name:20} | RGB{expected_tuple} | NOT FOUND")
                
                # Find closest color
                closest_distance = float('inf')
                closest_color = None
                
                for actual_color, count in color_counts:
                    distance = np.sqrt(sum((a - b)**2 for a, b in zip(expected_color, actual_color)))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_color = actual_color
                
                if closest_color:
                    count = next(c[1] for c in color_counts if c[0] == closest_color)
                    percentage = (count / (mask_rgb.shape[0] * mask_rgb.shape[1])) * 100
                    print(f"   Closest: RGB{closest_color} | {count:7d} pixels ({percentage:5.1f}%) | Distance: {closest_distance:.1f}")
        
        # Visualization
        print("\n📊 Creating color visualization...")
        create_color_visualization(mask_rgb, expected_colors, class_names, mask_path)
        
    except Exception as e:
        print(f"Error analyzing mask: {e}")

def create_color_visualization(mask_rgb, expected_colors, class_names, mask_path):
    """Create a visualization of the color analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original mask
    axes[0, 0].imshow(mask_rgb)
    axes[0, 0].set_title('Original Mask')
    axes[0, 0].axis('off')
    
    # Expected colors legend
    ax = axes[0, 1]
    for i, (class_name, color) in enumerate(zip(class_names.values(), expected_colors.values())):
        rect = plt.Rectangle((0, i), 1, 1, facecolor=np.array(color)/255.0, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.1, i + 0.5, f"{class_name} - RGB{tuple(color)}", va='center', fontsize=10)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(class_names))
    ax.set_title('Expected Colors (from classes.json)')
    ax.axis('off')
    
    # Actual colors found
    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    ax = axes[1, 0]
    
    for i, color in enumerate(unique_colors[:10]):  # Show first 10 colors
        rect = plt.Rectangle((0, i), 1, 1, facecolor=color/255.0, edgecolor='black')
        ax.add_patch(rect)
        
        count = np.sum(np.all(mask_rgb == color, axis=-1))
        percentage = (count / (mask_rgb.shape[0] * mask_rgb.shape[1])) * 100
        
        ax.text(1.1, i + 0.5, f"RGB{tuple(color)} ({percentage:.1f}%)", va='center', fontsize=10)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, min(10, len(unique_colors)))
    ax.set_title('Actual Colors Found in Mask')
    ax.axis('off')
    
    # Color histogram
    ax = axes[1, 1]
    colors_flat = mask_rgb.reshape(-1, 3)
    
    # Count pixels for each unique color
    unique_colors, counts = np.unique(colors_flat, axis=0, return_counts=True)
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    top_colors = unique_colors[sorted_indices[:6]]  # Top 6 colors
    top_counts = counts[sorted_indices[:6]]
    
    bars = ax.bar(range(len(top_counts)), top_counts, 
                  color=[c/255.0 for c in top_colors])
    
    ax.set_title('Top Colors by Pixel Count')
    ax.set_xlabel('Color Index')
    ax.set_ylabel('Pixel Count')
    
    # Add RGB values as labels
    for i, (color, count) in enumerate(zip(top_colors, top_counts)):
        ax.text(i, count + max(top_counts)*0.01, f'RGB{tuple(color)}', 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.suptitle(f'Color Analysis: {os.path.basename(mask_path)}', fontsize=16)
    plt.tight_layout()
    plt.savefig('color_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Color analysis visualization saved as 'color_analysis.png'")

def suggest_fixes(mask_dir, classes_json_path='./dubai_dataset/classes.json'):
    """Suggest fixes based on analysis of multiple masks"""
    
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 50)
    
    # Analyze a few masks
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')][:3]
    
    all_colors = set()
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
            all_colors.update(tuple(c) for c in unique_colors)
    
    print(f"Analyzed {len(mask_files)} masks")
    print(f"Total unique colors found: {len(all_colors)}")
    
    if len(all_colors) > 10:
        print("❌ Too many unique colors - masks might be RGB photos instead of indexed masks")
        print("   Fix: Ensure masks are properly labeled/indexed images")
    elif len(all_colors) <= 6:
        print("✅ Reasonable number of colors found")
        print("   Fix: Update classes.json with actual colors or remap colors")
    
    print("\n💡 RECOMMENDED ACTIONS:")
    print("1. Run this diagnostic on 2-3 different mask files")
    print("2. Check if your masks are indexed color images vs RGB images")
    print("3. Update classes.json colors to match actual mask colors")
    print("4. Or create a color remapping function")

def main():
    """Run the color diagnostic"""
    
    # Example usage
    mask_path = './dubai_dataset/masks/image_part_001.png'  # Update this path
    classes_json_path = './dubai_dataset/classes.json'
    
    print("🚀 Running Dubai Dataset Color Diagnostic...")
    print("Update the mask_path variable to point to your actual mask file")
    print()
    
    if os.path.exists(mask_path):
        diagnose_color_issues(mask_path, classes_json_path)
    else:
        print(f"❌ Mask file not found: {mask_path}")
        print("Please update the mask_path variable in this script")
        
        # Try to find mask files
        mask_dir = './dubai_dataset/masks/'
        if os.path.exists(mask_dir):
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
            if mask_files:
                print(f"\nFound {len(mask_files)} mask files:")
                for i, f in enumerate(mask_files[:5]):
                    print(f"  {i+1}: {f}")
                print(f"\nTry running with: {os.path.join(mask_dir, mask_files[0])}")
    
    # Suggest fixes
    mask_dir = './dubai_dataset/masks/'
    if os.path.exists(mask_dir):
        suggest_fixes(mask_dir, classes_json_path)

if __name__ == "__main__":
    main()
