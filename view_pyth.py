import os
import kagglehub
from torchvision import datasets
from src.planning.config import EMOTIONS


def locate_split_dirs(base_path):
    train_dir = None
    test_dir = None
    for root, dirs, _ in os.walk(base_path):
        if "train" in dirs:
            train_dir = os.path.join(root, "train")
        if "test" in dirs:
            test_dir = os.path.join(root, "test")

    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Could not locate 'train' or 'test' directories in {base_path}")
    return train_dir, test_dir


class ImageFolderIgnoreClass(datasets.ImageFolder):
    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        classes = [cls for cls in classes if EMOTIONS.__contains__(cls)]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def main():
    data_path = kagglehub.dataset_download("subhaditya/fer2013plus")
    train_dir, test_dir = locate_split_dirs(data_path)

    train_set = ImageFolderIgnoreClass(train_dir)
    test_set = ImageFolderIgnoreClass(test_dir)

    print("Dataset cache path:", data_path)
    print("Train directory:", train_dir)
    print("Test directory:", test_dir)
    print("Train emotions:", train_set.classes)
    print("Test emotions:", test_set.classes)
    print("Train class->index:", train_set.class_to_idx)
    print("Test class->index:", test_set.class_to_idx)
    print("Class lists match:", train_set.classes == test_set.classes)
    print("Class mappings match:", train_set.class_to_idx == test_set.class_to_idx)


if __name__ == "__main__":
    main()