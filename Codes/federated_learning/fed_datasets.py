import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from preprocessing import read_dicom, preprocess_image

CLASS_MAP = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
DENSITY_CLASSES = ["DENSITY A", "DENSITY B", "DENSITY C", "DENSITY D"]

def _transform(use_augmentation):
  if use_augmentation:
    return transforms.Compose([
      transforms.RandomRotation(degrees=15),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ColorJitter(brightness=0.1, contrast=0.1),
      transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
  return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
  ])

def _to_pil(image_array):
  if len(image_array.shape) == 3:
    image_array = image_array[:, :, 0]
  elif len(image_array.shape) > 3:
    image_array = image_array[0, :, :, 0] if len(image_array.shape) == 4 else image_array.squeeze()
  return Image.fromarray(image_array.astype(np.uint8), mode='L')

class VinDrCCDataset(Dataset):
  # Dataset VinDr-Mammo (vue CC)
  def __init__(self, annotations_csv, image_root, split='training', use_augmentation=False):
    df = pd.read_csv(annotations_csv)
    df = df[(df['split'] == split) & (df['view_position'] == 'CC')].reset_index(drop=True)
    df['label'] = df['breast_density'].map(CLASS_MAP)
    self.df = df
    self.image_root = image_root
    self.split = split
    self.density_map = {"DENSITY A": "density_A", "DENSITY B": "density_B", "DENSITY C": "density_C", "DENSITY D": "density_D"}
    self.split_folder = {"training": "train", "test": "test"}
    self.transform = _transform(use_augmentation)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    study_id, image_id = row['study_id'], row['image_id']
    density = self.density_map.get(row["breast_density"])
    split_folder = self.split_folder.get(row["split"], "train")

    image_path = os.path.join(self.image_root, split_folder, density, study_id, f"{image_id}.dicom")
    if not os.path.exists(image_path):
      image_path = os.path.join(self.image_root, 'images', study_id, f"{image_id}.dicom")
      if not os.path.exists(image_path):
        image_path = os.path.join(self.image_root, study_id, f"{image_id}.dicom")

    try:
      image = read_dicom(image_path)
      image = preprocess_image(image, laterality=row["laterality"])
      image = self.transform(_to_pil(image))
      return image, row['label']
    except Exception:
      return torch.randn(1, 224, 224), 0

class DDSMDataset(Dataset):
  # Dataset DDSM/CBIS-DDSM (vue CC)
  def __init__(self, annotations_csv, split='training', use_augmentation=False):
    df = pd.read_csv(annotations_csv)
    df = df[df['split'] == split].reset_index(drop=True)
    df['label'] = df['breast_density'].map(CLASS_MAP)
    self.df = df
    self.transform = _transform(use_augmentation)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    try:
      image = read_dicom(row['image_path'])
      image = preprocess_image(image, laterality=row["laterality"])
      image = self.transform(_to_pil(image))
      return image, row['label']
    except Exception:
      return torch.randn(1, 224, 224), 0

class INbreastDataset(Dataset):
  # Dataset INbreast
  def __init__(self, annotations_csv, split='training', use_augmentation=False):
    df = pd.read_csv(annotations_csv)
    df = df[df['split'] == split].reset_index(drop=True)
    df['label'] = df['breast_density'].map(CLASS_MAP)
    self.df = df
    self.transform = _transform(use_augmentation)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    try:
      image = read_dicom(row['image_path'])
      image = preprocess_image(image, laterality=row["laterality"])
      image = self.transform(_to_pil(image))
      return image, row['label']
    except Exception:
      return torch.randn(1, 224, 224), 0

def _build_cc_mlo_pairs(df):
  # Regrouper en paires CC+MLO pour le même sein
  pairs = []
  for (study_id, laterality), group in df.groupby(['study_id', 'laterality']):
    cc_rows = group[group['view_position'] == 'CC']
    mlo_rows = group[group['view_position'] == 'MLO']
    if cc_rows.empty or mlo_rows.empty:
      continue
    cc_row, mlo_row = cc_rows.iloc[0], mlo_rows.iloc[0]
    pairs.append({
      'study_id': study_id,
      'laterality': laterality,
      'cc_row': cc_row,
      'mlo_row': mlo_row,
      'label': cc_row['label'],
    })
  return pairs

class VinDrPairedDataset(Dataset):
  # Dataset VinDr apparié CC+MLO
  def __init__(self, annotations_csv, image_root, split='training', use_augmentation=False):
    df = pd.read_csv(annotations_csv)
    df = df[df['split'] == split].copy()
    df['label'] = df['breast_density'].map(CLASS_MAP)
    self.pairs = _build_cc_mlo_pairs(df)
    self.image_root = image_root
    self.split = split
    self.density_map = {"DENSITY A": "density_A", "DENSITY B": "density_B", "DENSITY C": "density_C", "DENSITY D": "density_D"}
    self.split_folder = {"training": "train", "test": "test"}
    self.transform = _transform(use_augmentation)

  def __len__(self):
    return len(self.pairs)

  def _load(self, row):
    study_id, image_id = row['study_id'], row['image_id']
    density = self.density_map.get(row["breast_density"])
    split_folder = self.split_folder.get(row["split"], "train")
    image_path = os.path.join(self.image_root, split_folder, density, study_id, f"{image_id}.dicom")
    if not os.path.exists(image_path):
      image_path = os.path.join(self.image_root, 'images', study_id, f"{image_id}.dicom")
      if not os.path.exists(image_path):
        image_path = os.path.join(self.image_root, study_id, f"{image_id}.dicom")
    image = read_dicom(image_path)
    image = preprocess_image(image, laterality=row["laterality"])
    return self.transform(_to_pil(image))

  def __getitem__(self, idx):
    pair = self.pairs[idx]
    try:
      cc_tensor = self._load(pair['cc_row'])
      mlo_tensor = self._load(pair['mlo_row'])
      return cc_tensor, mlo_tensor, pair['label']
    except Exception:
      return torch.randn(1, 224, 224), torch.randn(1, 224, 224), 0

class DDSMPairedDataset(Dataset):
  # Dataset DDSM apparié CC+MLO
  def __init__(self, annotations_csv, split='training', use_augmentation=False):
    df = pd.read_csv(annotations_csv)
    df = df[df['split'] == split].copy()
    df['label'] = df['breast_density'].map(CLASS_MAP)
    self.pairs = _build_cc_mlo_pairs(df)
    self.transform = _transform(use_augmentation)

  def __len__(self):
    return len(self.pairs)

  def _load(self, row):
    image = read_dicom(row['image_path'])
    image = preprocess_image(image, laterality=row["laterality"])
    return self.transform(_to_pil(image))

  def __getitem__(self, idx):
    pair = self.pairs[idx]
    try:
      cc_tensor = self._load(pair['cc_row'])
      mlo_tensor = self._load(pair['mlo_row'])
      return cc_tensor, mlo_tensor, pair['label']
    except Exception:
      return torch.randn(1, 224, 224), torch.randn(1, 224, 224), 0
