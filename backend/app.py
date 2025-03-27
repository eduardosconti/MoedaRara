from flask import Flask, request, jsonify
from flask_cors import CORS  # Para habilitar CORS
import cv2
import numpy as np
import base64
import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelos"))

from coinClassifier import model  # classificador
from oneClass import oc_svm, scaler, mean_normal_image  # objetos já treinados

app = Flask(__name__)

# Habilitando CORS para aceitar requisições de qualquer origem (para testes)
CORS(app)

@app.route('/classificar', methods=['POST'])
def classificar():
    # Recebe a imagem enviada no campo 'image'
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Pré-processamento para o classificador
    img_resized = cv2.resize(img, (128, 128))
    # Converter para tensor (necessário para o modelo)
    img_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    
    # Previsão com o modelo (classificador)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    classe = int(predicted.item())
    
    # Processamento com One-Class SVM
    img_oc = cv2.resize(img, (256, 256)).flatten().reshape(1, -1)
    img_oc = scaler.transform(img_oc)
    pred_oc = oc_svm.predict(img_oc)[0]
    status = "Normal" if pred_oc == 1 else "Anômala"
    
    # Gerar o mapa de diferença se for anômala
    diff_b64 = ""
    if pred_oc == -1:
        img_256 = cv2.resize(img, (256, 256))
        diff = cv2.absdiff(img_256, mean_normal_image.astype(np.uint8))
        _, buffer = cv2.imencode('.png', diff)
        diff_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Retornar a resposta em formato JSON
    return jsonify({
        'classe_moeda': classe,
        'status': status,
        'diff_map': diff_b64
    }), 200  # Código de sucesso

if __name__ == '__main__':
    app.run(debug=True)