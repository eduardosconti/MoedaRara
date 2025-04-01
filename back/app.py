from flask import Flask, request, jsonify
import os
import base64
import modelo.oneClass as oneClass  # Importando seu script oneClass.py da pasta modelo

app = Flask(__name__)

@app.route('/processar', methods=['POST'])
def processar_imagem():
    data = request.json['image']
    
    # Define o caminho da imagem capturada, dentro de RA/imagens/
    image_path = os.path.join('RA', 'imagens', 'temp.png')
    
    # Cria a pasta de imagens se não existir
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))

    # Salva a imagem enviada pelo frontend
    with open(image_path, "wb") as img_file:
        img_file.write(base64.b64decode(data.split(",")[1]))

    # Executa a detecção de anomalias
    output_path = oneClass.detectar_anomalias(image_path)  # Chama a função do seu script

    # Se o heatmap foi gerado, retorna o caminho para ele
    if output_path:
        # O heatmap é salvo na pasta RA/resultados/
        return jsonify({"heatmap_url": output_path})
    else:
        return jsonify({"error": "Nenhuma anomalia detectada."}), 400

if __name__ == "__main__":
    app.run(debug=True)
