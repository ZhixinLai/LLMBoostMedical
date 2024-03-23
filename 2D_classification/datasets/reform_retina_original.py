import pandas as pd
import shutil
import os

df = pd.read_csv('./DeepDRiD/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv')

for index, row in df.iterrows():
    label_path = row['image_path'].replace("\\", "/")
    path_parts = label_path.strip("/").split("/")
    last_two_parts = "/".join(path_parts[-2:])
    image_path = './DeepDRiD/regular_fundus_images/regular-fundus-validation/Images/' + last_two_parts
    patient_DR_Level = row['patient_DR_Level']

    target_dir = f'./DeepDRiD_RetinaMNIST/test/{patient_DR_Level}/'

    os.makedirs(target_dir, exist_ok=True)

    target_image_path = os.path.join(target_dir, os.path.basename(image_path))
    print(image_path)
    print(target_image_path)

    shutil.copy(image_path, target_image_path)
