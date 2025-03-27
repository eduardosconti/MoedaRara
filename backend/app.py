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

def resize_with_pad(image, target_size):
    """Redimensiona mantendo aspect ratio com preenchimento preto"""
    h, w = image.shape
    scale = min(target_size[0]/w, target_size[1]/h)
    resized = cv2.resize(image, (int(w*scale), int(h*scale)))
    pad_top = (target_size[1] - resized.shape[0]) // 2
    pad_bottom = target_size[1] - resized.shape[0] - pad_top
    pad_left = (target_size[0] - resized.shape[1]) // 2
    pad_right = target_size[0] - resized.shape[1] - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                            cv2.BORDER_CONSTANT, value=0)

@app.route('/classificar', methods=['POST'])
def classificar():
    # Verifica se a imagem foi enviada
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['image']
    
    # Verifica se o arquivo tem nome
    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo vazio'}), 400
    
    try:
        # Lê a imagem
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Verificação de dimensões e validade da imagem
        if img is None:
            return jsonify({'error': 'Formato de imagem inválido'}), 400
        
        if img.size == 0:
            return jsonify({'error': 'Imagem vazia'}), 400
            
        # Verifica tamanho mínimo (50x50 pixels)
        if img.shape[0] < 50 or img.shape[1] < 50:
            return jsonify({'error': 'Imagem muito pequena (mínimo 50x50 pixels)'}), 400
        
        # --- NOVO PRÉ-PROCESSAMENTO ADICIONADO AQUI ---
        # 1. Redução de ruído
        img = cv2.medianBlur(img, 5)
        
        # 2. Binarização adaptativa
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 3. Redimensionamento com padding
        img_resized = resize_with_pad(img, (128, 128))
        # ----------------------------------------------
        
        # Verifica se o redimensionamento funcionou
        if img_resized.size == 0:
            return jsonify({'error': 'Falha no pré-processamento'}), 500
            
        # Converter para tensor
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
        
        # Gerar máscara de anomalias e imagem original redimensionada
        anomaly_mask_b64 = ""
        original_img_b64 = ""
        
        if pred_oc == -1:
            img_256 = cv2.resize(img, (256, 256))
            
            # Converter para RGB para compatibilidade com visualização
            img_256_rgb = cv2.cvtColor(img_256, cv2.COLOR_GRAY2RGB)
            diff = cv2.absdiff(img_256, mean_normal_image.astype(np.uint8))
            
            # Criar máscara vermelha
            anomaly_mask = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            anomaly_mask[:, :, 0] = 0  # Remove blue channel
            anomaly_mask[:, :, 1] = 0  # Remove green channel
            
            # Codificar imagens
            _, buffer_mask = cv2.imencode('.png', anomaly_mask)
            _, buffer_original = cv2.imencode('.png', img_256_rgb)
            
            anomaly_mask_b64 = base64.b64encode(buffer_mask).decode('utf-8')
            original_img_b64 = base64.b64encode(buffer_original).decode('utf-8')
        
        # Debug (opcional)
        print(f"Class: {classe}, Status: {status}, Mask Size: {len(anomaly_mask_b64)}, Original Size: {len(original_img_b64)}")
        
        return jsonify({
            'classe_moeda': classe,
            'status': status,
            'anomaly_mask': anomaly_mask_b64,
            'original_img': original_img_b64,
            'message': 'Processamento concluído com sucesso'
        }), 200
        
    except Exception as e:
        app.logger.error(f"Erro no processamento: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Erro interno no processamento',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)