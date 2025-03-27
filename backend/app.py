from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modelos"))

from coinClassifier import model
from oneClass import oc_svm, scaler, mean_normal_image

app = Flask(__name__)
CORS(app)

def smart_resize(image, target_size):
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
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['image']
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Pré-processamento
        img = cv2.medianBlur(img, 5)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img_resized = smart_resize(img, (128, 128))
        
        # Classificação
        img_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
        classe = int(predicted.item())
        
        # Detecção de anomalias
        img_oc = cv2.resize(img, (256, 256)).flatten().reshape(1, -1)
        img_oc = scaler.transform(img_oc)
        pred_oc = oc_svm.predict(img_oc)[0]
        status = "Normal" if pred_oc == 1 else "Anômala"
        
        # Gerar máscara
        anomaly_mask_b64 = ""
        original_img_b64 = ""
        
        if pred_oc == -1:
            img_256 = smart_resize(img, (256, 256))
            img_256_rgb = cv2.cvtColor(img_256, cv2.COLOR_GRAY2RGB)
            diff = cv2.absdiff(img_256, mean_normal_image.astype(np.uint8))
            
            anomaly_mask = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            anomaly_mask[:, :, 0:2] = 0  # Mantém apenas vermelho
            
            _, buffer_mask = cv2.imencode('.png', anomaly_mask)
            _, buffer_original = cv2.imencode('.png', img_256_rgb)
            
            anomaly_mask_b64 = base64.b64encode(buffer_mask).decode('utf-8')
            original_img_b64 = base64.b64encode(buffer_original).decode('utf-8')

        return jsonify({
            'classe_moeda': classe,
            'status': status,
            'anomaly_mask': anomaly_mask_b64,
            'original_img': original_img_b64
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)