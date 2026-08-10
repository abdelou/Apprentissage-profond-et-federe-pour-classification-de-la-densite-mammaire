#%%
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from preprocessing import read_dicom, preprocess_image, normalize_intensity_global
from PIL import Image
#%% Charger le fichier DICOM
root = 'G:/MA1/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/output/train/density_A'
dossier = '5051671fd96995d0b3248f6eb7e8a2bc'
image_id = '2a74ba51e0db4679a7cb6ec34ddc1250.dicom'
path = root + '/' + dossier + '/' + image_id
ds = pydicom.dcmread(path)
laterality = 'R'
image = read_dicom(path)
image = preprocess_image(image, laterality=laterality)
#%%ds = pydicom.dcmread(path)
img = ds.pixel_array.astype(np.float32)
    
    # Gestion de la photométrie DICOM
photometric_interpretation = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    
    # Inversion si nécessaire pour avoir un fond noir cohérent
if photometric_interpretation == 'MONOCHROME1':
        # Fond blanc -> Fond noir (inversion)
    img = ds.BitsAllocated - img - 1
    print(f"[DICOM] Image inversée (MONOCHROME1): {path}")
elif photometric_interpretation == 'MONOCHROME2':
        # Déjà en fond noir, pas d'inversion nécessaire
    print(f"[DICOM] Image normale (MONOCHROME2): {path}")
else:
    print(f"[DICOM] Photométrie inconnue: {photometric_interpretation}")
# Afficher toutes les métadonnées
#print(ds)

# %%
img1 = cv2.resize(img, (224, 224))
plt.imshow(img1, cmap=plt.cm.gray)
plt.title("Image DICOM")
plt.axis("off")
plt.show()
#%%

#%%
#img = cv2.resize(img, (224, 224)
img_norm = normalize_intensity_global(img)
img_pil = Image.fromarray(img_norm).convert("L")  # "L" = niveaux de gris ; mettre "RGB" si color

# Appliquer les augmentations
augmentations = transforms.Compose([
    transforms.RandomRotation(degrees=2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])
img_np = augmentations(img_pil)
img_np = np.array(img_np) / 255.0
plt.imshow(img_np, cmap=plt.cm.gray)
plt.title("Image DICOM")
plt.axis("off")
plt.show()

# %%
img = ds.pixel_array

# Définir le nombre de pixels à rogner sur chaque bord
crop_top = 50      # nombre de pixels à couper en haut
crop_bottom = 50   # en bas
crop_left = 10     # à gauche
crop_right = 700    # à droite
img = img - np.min(img)
img = img / (np.max(img) + 1e-8)
img = (img * 255).astype(np.uint8)
# Rogner l'image
cropped_img = img[crop_top:img.shape[0]-crop_bottom, crop_left:img.shape[1]-crop_right]
mask = cropped_img > 10
cropped_img = cropped_img * mask

# Afficher l'image rognée
plt.imshow(img[:, :-700], cmap=plt.cm.gray)
plt.title("Image DICOM sans annotations")
plt.axis("off")
plt.show()
# %%
root = 'G:/MA1/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/output/train/density_D'
dossier = '00bed569a272c750cf03b561886a1489'
image_id = 'cf71975732419786732a6c0f9c106868.dicom'
path = root + '/' + dossier + '/' + image_id
ds = pydicom.dcmread(path)
img = ds.pixel_array
img = preprocess_image(img, laterality= 'L')
# Afficher l'image et l'histogramme
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Image
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Image de mammographie densité D")
axes[0].axis('off')

# Histogramme des niveaux de gris
#ne prendre les valeurs entre 40 et 200
mask = (img.ravel() > 100) & (img.ravel() < 300)
axes[1].hist(img.ravel()[mask], bins=256, color='gray')
axes[1].set_title("Répartition des niveaux de gris (entre 100 et 300)")
axes[1].set_xlabel("Valeurs de pixels")
axes[1].set_ylabel("Nombre de pixels")

plt.tight_layout()
plt.show()
# %%
