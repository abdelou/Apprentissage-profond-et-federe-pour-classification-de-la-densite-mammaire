import torch
print("CUDA dispo ?", torch.cuda.is_available())
print("Nombre de GPU :", torch.cuda.device_count())
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun")
