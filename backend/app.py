import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

app = Flask(__name__)

# Configuração de pastas para salvar imagens recebidas
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Carregar modelo treinado de detecção de anomalias
scaler = StandardScaler()
oc_svm = OneClassSVM(kernel="rbf", gamma=0.001, nu=0.01)

# Simulação de treinamento (substitua pelos seus dados reais)
def train_anomaly_model():
    print("Treinando modelo de detecção de anomalias...")
    # Carregar imagens de treinamento e normalizar
    sample_images = np.random.rand(50, 256*256)  # Simulação de dataset normalizado
    scaler.fit(sample_images)
    oc_svm.fit(scaler.transform(sample_images))
    print("Modelo treinado com sucesso!")

train_anomaly_model()

def generate_anomaly_heatmap(image_path):
    """
    Carrega uma imagem, aplica o modelo de anomalia e gera um mapa de calor.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img_flat = img.flatten().reshape(1, -1)

    # Normalizar a imagem antes de passar no modelo
    img_flat = scaler.transform(img_flat)
    prediction = oc_svm.predict(img_flat)[0]

    if prediction == 1:
        return None  # Imagem normal, sem mapa de calor

    # Criar mapa de calor baseado na diferença da média
    mean_image = np.mean(scaler.transform(np.random.rand(50, 256*256)), axis=0).reshape(256, 256)
    diff_image = np.abs(img.reshape(256, 256) - mean_image)

    heatmap_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path).replace(".jpg", "_heatmap.png"))

    # Gerar e salvar o heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(diff_image, cmap="hot", interpolation="nearest")
    plt.axis("off")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return heatmap_path

@app.route("/detect", methods=["POST"])
def detect_anomalies():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    print(f"Imagem recebida: {filename}")

    # Gera o mapa de calor
    heatmap_path = generate_anomaly_heatmap(image_path)

    if heatmap_path:
        return jsonify({"status": "Anomalia detectada", "heatmap": heatmap_path})
    else:
        return jsonify({"status": "Sem anomalias detectadas"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
