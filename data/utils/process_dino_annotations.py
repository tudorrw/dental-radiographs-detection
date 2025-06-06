import json
import os
from utils.mapper import ToothLabelMapper

label_mapper = ToothLabelMapper()

def process_dino_annotations(input_file, output_file):
    """
    Process DINO annotation file to map tooth labels to the correct FDI numbering and class indices.
    Args:
        input_file (str): Path to the input annotation file (e.g., instances_train2017.json)
        output_file (str): Path to save the processed annotation file
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    for ann in data['annotations']:
        print("ann", ann['image_id'])
        if 'category_id' in ann:
            category_id = ann['category_id']
            # Extract category_id_1 and category_id_2 using the formula category_id = category_id_1 * 8 + category_id_2
            category_id_1 = category_id // 8
            category_id_2 = category_id % 8
        
            tooth_id = category_id_1 * 10 + category_id_2 + 1
            mapped_class_id = int(label_mapper.encode([tooth_id])[0])
            ann['category_id'] = mapped_class_id - 1

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    base_dir = 'dataset/coco/dino/quadrant_enumeration/annotations'
    output_dir = 'utils'
    for split in ['val', 'test', 'train']:
        input_file = os.path.join(base_dir, f'instances_{split}2017.json')
        output_file = os.path.join(base_dir, f'instances_{split}2017_processed.json')
        process_dino_annotations(input_file, output_file)
        print(f"Processed {input_file} -> {output_file}") 