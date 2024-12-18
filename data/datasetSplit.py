import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a dataset into training, validation, and test sets.
    
    :param dataset_dir: Path to the directory containing class folders.
    :param output_dir: Path to the output directory for the split dataset.
    :param train_ratio: Proportion of data for the training set.
    :param val_ratio: Proportion of data for the validation set.
    :param test_ratio: Proportion of data for the test set.
    """
    # Ensure the ratios sum up to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Paths for output
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    # Create directories
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)

    # Process each class folder
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Gather all images in the class
        images = [f for f in os.listdir(class_path) if f.endswith(".png")]
        random.shuffle(images)

        # Calculate split sizes
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count

        # Print info
        print(f"Class: {class_name} | Train: {train_count}, Val: {val_count}, Test: {test_count}")

        # Paths for each split
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)

        # Create class directories in splits
        for path in [class_train_dir, class_val_dir, class_test_dir]:
            os.makedirs(path, exist_ok=True)

        # Split and move images
        for i, image in enumerate(images):
            src_path = os.path.join(class_path, image)
            if i < train_count:
                dest_path = os.path.join(class_train_dir, image)
            elif i < train_count + val_count:
                dest_path = os.path.join(class_val_dir, image)
            else:
                dest_path = os.path.join(class_test_dir, image)
            shutil.copy(src_path, dest_path)

    print("Dataset split completed successfully!")

if __name__ == "__main__":
    # Paths
    dataset_dir = "data/dataset"  # Input folder containing the 13 class folders
    output_dir = "data/split_dataset"  # Output folder for split dataset

    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Split the dataset
    split_dataset(dataset_dir, output_dir, train_ratio, val_ratio, test_ratio)