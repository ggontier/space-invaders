import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import asyncio
import websockets
from torchvision import transforms
from PIL import Image 

# Définition du modèle (correspond à l'entraînement)
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes = 4):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1) # 3x128x128 -> 16x128x128
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2) # 16x128x128 -> 16x64x64
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 16x64x64 -> 8x64x64
        self.fc1 = nn.Linear(8 * 64 * 64, 128) # 8x64x64 -> 128
        self.fc2 = nn.Linear(128, num_classes) # 128 -> 4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = x.view(-1, 8 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Charger le modèle
model = ImprovedCNN(num_classes=4)
state_dict = torch.load("improved_model.pth", map_location=torch.device("cpu"))

if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])  # Si sauvegardé avec un dictionnaire
else:
    model.load_state_dict(state_dict)  # Sinon, charger directement

model.eval()

# Définir les transformations pour prétraiter les images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Taille adaptée au CNN
    transforms.ToTensor(),  # Convertir en tenseur PyTorch
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # Normalisation
])

# Mapping des prédictions aux commandes Space Invaders
commands = ["ENTER", "FIRE", "LEFT", "RIGHT"]

# Fonction pour capturer et classer les mouvements
async def detect_and_send():
    uri = "ws://localhost:8765"
    print("🎮 Contrôle via webcam connecté à Space Invaders...")

    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # Ouvre la webcam

        if not cap.isOpened():
            print("❌ Impossible d'accéder à la webcam")
            return

        print("✅ Webcam activée ! Détection en cours...")
        
        last_command = None  # Permet d'éviter les doublons inutiles

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur lors de la capture de l'image")
                break

            # Convertir en format PIL
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV → PIL
            img_tensor = transform(img_pil).unsqueeze(0)  # Appliquer transformations

            # Prédiction
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                command = commands[predicted.item()]

            # Envoyer la commande uniquement si elle a changé
            print(f"Détection : {command}")
            if command != last_command:
                await websocket.send(command)
                print(f"🔹 Commande envoyée : {command}")
                last_command = command  # Mise à jour de la dernière commande envoyée

            # Afficher la capture vidéo avec la commande détectée
            cv2.putText(frame, f"Commande: {command}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Control", frame)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Contrôle arrêté.")
                

# Exécuter la boucle principale
if __name__ == "__main__":
    asyncio.run(detect_and_send())
