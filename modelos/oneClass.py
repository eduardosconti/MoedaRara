import os
import cv2
import numpy as np
import seaborn as sns
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

def load_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (256, 256))
        img = img.flatten().reshape(1, -1)
    return img

def save_image(image, filename):
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

moeda = "10" # 1 2 5 10 20 50 verso
base_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_path, "Cruzeiro_Novo", "output treino", moeda)
test_image_path = os.path.join(base_path, "Cruzeiro_Novo", "output teste", moeda)
save_path = os.path.join(base_path, "Cruzeiro_Novo", "output resultados", moeda)

if not os.path.exists(save_path):
    os.makedirs(save_path)

train_images = load_images_from_folder(train_path)
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.01)
oc_svm.fit(train_images)

mean_normal_image = np.mean(train_images, axis=0).reshape(256, 256)

for image_name in os.listdir(test_image_path):
    test_image = load_single_image(os.path.join(test_image_path, image_name))
    if test_image is not None:
        prediction = oc_svm.predict(test_image)[0]
        label = "Normal" if prediction == 1 else "Anomalous"
        test_image_reshaped = test_image.reshape(256, 256)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(test_image_reshaped, cmap='gray')
        plt.title(f"Test Image - {label}")
        save_image(plt, os.path.join(save_path, f"{image_name}_test.png"))
        
        if prediction == -1:
            diff_image = np.abs(test_image_reshaped - mean_normal_image)
            plt.figure()
            plt.imshow(diff_image, cmap='hot')
            plt.title("Diff Map")
            save_image(plt, os.path.join(save_path, f"{image_name}_diff.png"))
            
            anomaly_score = test_image_reshaped - mean_normal_image
            plt.figure()
            plt.imshow(anomaly_score, cmap='coolwarm', interpolation='nearest')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title("Anomaly Heatmap")
            save_image(plt, os.path.join(save_path, f"{image_name}_heatmap.png"))
    else:
        print(f"Error loading test image: {image_name}")
