import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
from utils.mapper import ToothLabelMapper

def load_voc_data(csv_path):
    """Load and process VOC format data from CSV."""
    df = pd.read_csv(csv_path)
    # Convert the annotations string to actual Python list of dictionaries
    df['annotations'] = df['annotations'].apply(ast.literal_eval)
    return df

def count_boxes_per_label(df):
    """Count number of boxes per label."""
    label_mapper = ToothLabelMapper()
    rows = []
    
    for _, row in df.iterrows():
        for ann in row['annotations']:
            category_id_1 = ann['category_id_1']
            category_id_2 = ann['category_id_2']
            tooth_id = category_id_1 * 10 + category_id_2 + 1
            mapped_class_id = int(label_mapper.encode([tooth_id])[0])
            rows.append({"image_id": row['id'], "class_id": mapped_class_id})
    
    ann_df = pd.DataFrame(rows)
    return ann_df['class_id'].value_counts().sort_index()

def plot_combined_distribution(train_counts, val_counts, test_counts, save_path=None):
    """Plot combined distribution of boxes across all datasets."""
    plt.style.use('bmh')
    plt.figure(figsize=(15, 8))
    
    x = range(len(train_counts))
    width = 0.25
    
    plt.bar([i - width for i in x], train_counts.values, width, label='Train', color='skyblue')
    plt.bar(x, val_counts.values, width, label='Val', color='lightgreen')
    plt.bar([i + width for i in x], test_counts.values, width, label='Test', color='salmon')
    
    plt.title('Combined Box Distribution Across All Datasets')
    plt.xlabel('Tooth Label')
    plt.ylabel('Number of Boxes')
    plt.xticks(x, train_counts.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def calculate_dataset_statistics():
    """Calculate and display statistics for the dataset splits."""
    # Define paths
    base_path = "dataset/pascal_voc/quadrant_enumeration"
    train_path = os.path.join(base_path, "quadrant_enumeration_voc_train.csv")
    val_path = os.path.join(base_path, "quadrant_enumeration_voc_val.csv")
    test_path = os.path.join(base_path, "quadrant_enumeration_voc_test.csv")
    
    # Load data
    train_df = load_voc_data(train_path)
    val_df = load_voc_data(val_path)
    test_df = load_voc_data(test_path)
    
    # Calculate box counts per label
    train_counts = count_boxes_per_label(train_df)
    val_counts = count_boxes_per_label(val_df)
    test_counts = count_boxes_per_label(test_df)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Train: {len(train_df)} unique images, {sum(train_counts)} bboxes")
    print(f"Val: {len(val_df)} unique images, {sum(val_counts)} bboxes")
    print(f"Test: {len(test_df)} unique images, {sum(test_counts)} bboxes")
    
    print("\nBox Distribution per Tooth Label:")
    print("\nTraining Set:")
    for label, count in train_counts.items():
        print(f"Tooth {label}: {count} boxes")
    
    print("\nValidation Set:")
    for label, count in val_counts.items():
        print(f"Tooth {label}: {count} boxes")
    
    print("\nTest Set:")
    for label, count in test_counts.items():
        print(f"Tooth {label}: {count} boxes")
    
    # Create visualization
    plot_combined_distribution(
        train_counts, 
        val_counts, 
        test_counts,
        save_path='box_distribution_existing_splits.png'
    )

if __name__ == "__main__":
    calculate_dataset_statistics() 