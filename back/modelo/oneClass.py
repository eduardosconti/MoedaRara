import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 🔹 Função para carregar imagens e treiná-las
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            img = img.flatten()
            images.append(img)
    return np.array(images)

# 🔹 Função para carregar uma única imagem
def load_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (256, 256))
        img = img.flatten().reshape(1, -1)
    return img

# 🔹 Função para salvar as imagens geradas
def save_image(image, filename):
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# 🔹 Caminhos do treinamento
moeda = "10"  # Pode mudar para outras moedas
base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "Cruzeiro_Novo", "output treino", moeda)

# 🔹 Carrega e treina o modelo OneClass SVM
train_images = load_images_from_folder(train_path)
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.01)
oc_svm.fit(train_images)
mean_normal_image = np.mean(train_images, axis=0).reshape(256, 256)

# 🔹 Função para processar uma única imagem e gerar um heatmap
def detectar_anomalias(image_path):
    test_image = load_single_image(image_path)
    if test_image is not None:
        prediction = oc_svm.predict(test_image)[0]
        label = "Normal" if prediction == 1 else "Anomalous"
        test_image_reshaped = test_image.reshape(256, 256)

        # 🔹 Se for anomalia, gera o heatmap
        if prediction == -1:
            anomaly_score = test_image_reshaped - mean_normal_image
            plt.figure()
            plt.imshow(anomaly_score, cmap='coolwarm', interpolation='nearest')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title("Anomaly Heatmap")

            # O heatmap será salvo em RA/resultados/
            output_path = os.path.join('RA', 'resultados', 'heatmap.png')
            save_image(plt, output_path)
            return output_path

    return None  # Se a imagem for normal, retorna None
