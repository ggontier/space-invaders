import cv2
import numpy as np
import torch
import torch.nn as nn
import asyncio
import websockets
import mediapipe as mp
import time
import threading
import os

# Désactiver les logs inutiles
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Charger le modèle entraîné (en PyTorch)
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialiser MediaPipe pour l'extraction des landmarks
mp_hands = mp.solutions.hands.Hands()

# Fonction pour extraire les landmarks d'une frame
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        landmarks = []
        for landmark in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])  # Ajouter les coordonnées
        return np.array(landmarks)
    else:
        return None

# Charger le modèle et ses poids
model = MLP(input_size = 63, num_classes = 4)  # Assurez-vous que input_size correspond à vos données
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()  # Mettre le modèle en mode évaluation

# Mapping des gestes prédits aux commandes
gesture_to_command = {
    0: "ENTER",   # classe 0 = entrée / start
    1: "FIRE",    # classe 1 = tirer
    2: "LEFT",    # classe 2 = gauche
    3: "RIGHT"    # classe 3 = droite
}

# Fonction pour afficher la vidéo dans un thread séparé
def show_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Space Invaders Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video thread...")
            break
    cap.release()
    cv2.destroyAllWindows()

# Fonction principale asynchrone pour envoyer des commandes
async def send_command(cap, websocket):
    last_frame_time = 0

    while True:
        # Lire une frame vidéo
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire une frame vidéo")
            break

        # Limiter la fréquence de traitement
        if time.time() - last_frame_time < 0.5:
            await asyncio.sleep(0.1)
            continue
        last_frame_time = time.time()

        # Réduire la taille de la frame pour MediaPipe
        frame_resized = cv2.resize(frame, (128, 128))

        # Extraire les landmarks
        landmarks = extract_landmarks(frame_resized)
        if landmarks is None:
            print("⚠️ Aucun landmark détecté pour cette frame.")
            continue

        # Prédiction avec le modèle
        if landmarks.shape[0] == 63:
            input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predictions = model(input_tensor)
                predicted_class = torch.argmax(predictions, dim=1).item()

            # Mapper le geste à une commande
            if predicted_class in gesture_to_command:
                command = gesture_to_command[predicted_class]
                await websocket.send(command)  # Envoyer la commande
                print(f"Commande envoyée : {command}")

            # Affichage de la commande détectée
            cv2.putText(frame, f"Commande: {command}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Afficher la frame avec la commande
        cv2.imshow("Space Invaders Control", frame)

        # Quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting control module...")
            break

# Fonction principale
async def main():
    uri = "ws://localhost:8765"
    
    print("Control module connecting to Space Invaders...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Reading video stream...")

            # Initialiser la webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Erreur : Impossible d'accéder au flux vidéo")
                return

            # Lancer la fonction pour envoyer les commandes en temps réel
            await send_command(cap, websocket)

    except asyncio.TimeoutError:
        print("❌ Timeout lors de la connexion WebSocket")
    except websockets.ConnectionClosedError:
        print("❌ Connexion WebSocket interrompue")

# Fonction principale
if __name__ == "__main__":
    asyncio.run(main())
