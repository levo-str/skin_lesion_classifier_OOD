import os

from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, data, img_dir, pixel_values_divider = 1, transform=None, target_transform=None):
        self.data = data
        self.img_labels = data['dx']
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_dict = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        self.max_value_img = pixel_values_divider

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['image_id'] + ".jpg")
        image = read_image(img_path) / self.max_value_img
        label = self.label_dict.get(self.data.iloc[idx]['dx'])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label  # , self.img_names.iloc[idx, 0]
