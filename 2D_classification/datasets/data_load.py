import os
# from medmnist import BreastMNIST, DermaMNIST, OrganAMNIST, ChestMNIST, RetinaMNIST, OCTMNIST, BloodMNIST, TissueMNIST, PneumoniaMNIST, OrganCMNIST, OrganSMNIST
from torchvision import transforms
from PIL import Image
import numpy as np
import medmnist
from medmnist import INFO
import argparse

def convert_label(label_str):
    return int(label_str.strip('[]'))

def load_dataset(data_flag):

    # train_dataset = OrganSMNIST(split="train", download=True)
    # val_dataset = OrganSMNIST(split="val", download=True)
    # test_dataset = OrganSMNIST(split="test", download=True)

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    train_dataset = DataClass(split='train', download=True)
    val_dataset = DataClass(split='val', download=True)
    test_dataset = DataClass(split='test', download=True)


    return train_dataset, val_dataset, test_dataset

def get_label_for_chest_mnist(label):
    arr = np.array(label)
    if np.all(arr == 0):
        return 0
    else:
        return np.argmax(arr == 1) + 1

def save_dataset(dataset, root_dir, split):
    to_pil = transforms.ToPILImage()

    for idx, (image, label) in enumerate(dataset):
        label = np.array(label)[0]

        # label = get_label_for_chest_mnist(np.array(label))

        label_dir = os.path.join(root_dir, split, str(label))
        os.makedirs(label_dir, exist_ok=True)

        if not isinstance(image, Image.Image):
            image = to_pil(image)

        image_path = os.path.join(label_dir, f"{idx}.jpg")

        image.save(image_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Dataloader script', add_help=False)
    parser.add_argument('--data_flag', default='retinamnist', type=str)

    args = parser.parse_args()

    data_flag = args.data_flag

    train_dataset, val_dataset, test_dataset = load_dataset(data_flag)

    first_sample, first_label = train_dataset[0]
    transform_to_tensor = transforms.ToTensor()
    tensor_image = transform_to_tensor(first_sample)
    print(f"First sample shape: {tensor_image.shape}, First label: {first_label}")

    save_dataset(train_dataset, f"./datasets/{data_flag}", "train")
    save_dataset(val_dataset, f"./datasets/{data_flag}", "val")
    save_dataset(test_dataset, f"./datasets/{data_flag}", "test")
