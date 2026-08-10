import torch
import argparse

def inspect_checkpoint(checkpoint_path):
    """
    Inspecte et affiche les informations d'un fichier checkpoint PyTorch.
    
    Args:
        checkpoint_path (str): Chemin vers le fichier checkpoint à inspecter
    """
    # Charger le checkpoint
    print(f"\nInspection du checkpoint: {checkpoint_path}")
    print("="*50)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        print("=== Clés du checkpoint ===")
        print(list(checkpoint.keys()))
        print()

        # Afficher les infos principales si elles existent
        print("=== Informations principales ===")
        if 'backbone_name' in checkpoint:
            print(f"Nom de la backbone : {checkpoint['backbone_name']}")
        if 'epoch' in checkpoint:
            print(f"Epoch sauvegardée : {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"Meilleure loss de validation : {checkpoint['best_val_loss']}")
        if 'model_state_dict' in checkpoint:
            print(f"Paramètres du modèle : {len(checkpoint['model_state_dict'])} tensors")
        if 'optimizer_state_dict' in checkpoint:
            print(f"Paramètres de l'optimiseur : {len(checkpoint['optimizer_state_dict'])} tensors")
        if 'scheduler_state_dict' in checkpoint:
            print("Scheduler présent dans le checkpoint.")
        if 'train_loss_history' in checkpoint:
            print(f"Historique des pertes d'entraînement : {len(checkpoint['train_loss_history'])} valeurs")
        if 'best_accuracy' in checkpoint:
            print(f"Meilleure précision : {checkpoint['best_accuracy']}")
        if 'val_loss_history' in checkpoint:
            print(f"Historique des pertes de validation : {len(checkpoint['val_loss_history'])} valeurs")
            
    except Exception as e:
        print(f"Erreur lors du chargement du checkpoint: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Inspecte un fichier checkpoint PyTorch")
    parser.add_argument("checkpoint_path", help="Chemin vers le fichier checkpoint à inspecter")
    
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    main()

# Pour voir la structure exacte d'une clé (exemple : model_state_dict)
# print(checkpoint['model_state_dict'].keys())