
# --- Imports ---
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from PIL import Image, ImageTk
import numpy as np
import os
from hybrid_model import HybridMammographyClassifier, ImageBranch
from preprocessing import read_dicom, preprocess_image

# --- Config ---
CONFIG = {
    'CLASS_MAP': {0: "DENSITY A", 1: "DENSITY B", 2: "DENSITY C", 3: "DENSITY D"},
}


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédiction de densité mammaire")
        self.model = None
        self.model_type = None
        self.image_path = None
        self.selected_model_dir = tk.StringVar()
        self.model_dirs = [
            "hybrid 2 branches images",
            "hybrid train cc rexnet",
            "hybrid train mlo resnet",
            "hybrid train mlo rexnet"
        ]
        tk.Label(root, text="Choisir le dossier du modèle hybride :").pack(pady=5)
        self.model_menu = tk.OptionMenu(root, self.selected_model_dir, *self.model_dirs)
        self.model_menu.pack(pady=5)
        self.img_label = tk.Label(root)
        self.img_label.pack()
        self.result_label = tk.Label(root, text="Classe prédite : ", font=("Arial", 14))
        self.result_label.pack(pady=10)
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Charger modèle", command=self.load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Sélectionner image/DICOM", command=self.select_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Prédire", command=self.predict).pack(side=tk.LEFT, padx=5)

    def load_model(self):
        model_dir = self.selected_model_dir.get()
        if not model_dir:
            messagebox.showwarning("Modèle", "Veuillez choisir un dossier de modèle hybride.")
            return
        base_path = os.path.join(os.getcwd(), model_dir)
        model_path = filedialog.askopenfilename(initialdir=base_path, title="Choisir le fichier modèle hybride", filetypes=[("PyTorch", "*.pth")])
        if not model_path:
            return
        # Poids pour les architectures à deux branches
        resnet50_weights, rexnet150_weights = None, None
        if model_dir == "hybrid 2 branches images":
            resnet50_weights = filedialog.askopenfilename(initialdir=base_path, title="Choisir les poids ResNet50", filetypes=[("PyTorch", "*.pth")])
            rexnet150_weights = filedialog.askopenfilename(initialdir=base_path, title="Choisir les poids RexNet150", filetypes=[("PyTorch", "*.pth")])
            if not resnet50_weights or not rexnet150_weights:
                messagebox.showerror("Erreur", "Poids manquants pour les deux branches.")
                return
        try:
            # Détection automatique du type de modèle
            if model_dir == "hybrid train cc rexnet" or model_dir == "hybrid train mlo rexnet" or model_dir == "hybrid train mlo resnet":
                self.model_type = "hybrid_single"
                self.model = HybridMammographyClassifier(
                    backbone="cnn",
                    input_channels=1,
                    image_feature_dim=512,
                    num_classes=4,
                    dropout=0.3,
                    pretrained=True
                )
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            elif model_dir == "hybrid 2 branches images":
                self.model_type = "hybrid_double"
                self.model = HybridMammographyClassifier(
                    input_channels=1,
                    image_feature_dim=512,
                    num_classes=4,
                    dropout=0.3,
                    pretrained=True,
                    resnet50_weights=resnet50_weights,
                    rexnet150_weights=rexnet150_weights
                )
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                messagebox.showerror("Erreur", "Type de modèle non reconnu ou poids manquants.")
                return
            self.model.eval()
            messagebox.showinfo("Modèle", f"Modèle chargé depuis : {model_dir} (type : {getattr(self, 'model_type', 'inconnu')})")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle : {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Choisir une image ou DICOM", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.dicom")])
        if not file_path:
            return
        self.image_path = file_path
        try:
            img = self.load_and_preprocess(file_path, for_display=True)
            img = img.resize((224, 224))
            tk_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=tk_img)
            self.img_label.image = tk_img
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image : {e}")

    def predict(self):
        if self.model is None:
            messagebox.showwarning("Modèle", "Veuillez charger un modèle.")
            return
        try:
            if self.model_type == "hybrid_double":
                mlo_path = filedialog.askopenfilename(title="Image MLO (ResNet50)", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.dicom")])
                cc_path = filedialog.askopenfilename(title="Image CC (RexNet150)", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.dicom")])
                mlo_img = self.load_and_preprocess(mlo_path)
                cc_img = self.load_and_preprocess(cc_path)
                with torch.no_grad():
                    outputs = self.model(mlo_img, cc_img)
                    pred = outputs.argmax(dim=1).item()
                    classe = CONFIG['CLASS_MAP'].get(pred, str(pred))
                    self.result_label.config(text=f"Classe prédite : {classe}")
            else:
                img_path = filedialog.askopenfilename(title="Image à prédire", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.dicom")])
                img = self.load_and_preprocess(img_path)
                with torch.no_grad():
                    outputs = self.model(img)
                    pred = outputs.argmax(dim=1).item()
                    classe = CONFIG['CLASS_MAP'].get(pred, str(pred))
                    self.result_label.config(text=f"Classe prédite : {classe}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de prédire : {e}")

    def load_and_preprocess(self, path, for_display=False):
        """Unifie le prétraitement pour DICOM et images classiques."""
        if path.lower().endswith('.dicom'):
            img = read_dicom(path)
            img = preprocess_image(img)
            if len(img.shape) == 3:
                img = img[:, :, 0]
            if for_display:
                return Image.fromarray(img.astype(np.uint8), mode='L')
        else:
            img = Image.open(path).convert('L')
            if for_display:
                return img
            img = np.array(img, dtype=np.float32)
        if not for_display:
            img = img / 255.0
            img = (img - 0.485) / 0.229
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
        return img

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
