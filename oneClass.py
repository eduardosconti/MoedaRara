import os
import cv2
import numpy as np
import seaborn as sns
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to standard size
            img = img.flatten()  # Flatten image for SVM input
            images.append(img)
    return np.array(images)

def load_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        img = img.flatten().reshape(1, -1)  # Reshape for SVM input
    return img

# Define dataset paths
base_path = os.path.dirname(os.path.abspath(__file__))  # Replace with your actual base path
train_path = os.path.join(base_path, "MoedaRara", "Cruzeiro_Novo", "output treino", "10")
test_image_path = os.path.join(base_path, "MoedaRara", "Cruzeiro_Novo", "output teste", "10", "10 centavos 1967 frente (10).png")

# Load training images
train_images = load_images_from_folder(train_path)

# Train One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma=1.0, nu=0.1, shrinking=False, tol=0.00001)  # nu controls the outlier fraction
oc_svm.fit(train_images)

# Compute average normal image for difference visualization
mean_normal_image = np.mean(train_images, axis=0).reshape(128, 128)

# Test on a single image
test_image = load_single_image(test_image_path)
if test_image is not None:
    prediction = oc_svm.predict(test_image)[0]
    label = "Normal" if prediction == 1 else "Anomalous"
    
    test_image_reshaped = test_image.reshape(128, 128)
    
    # Show original test image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image_reshaped, cmap='gray')
    plt.title(f"Test Image - {label}")
    
    if prediction == -1:
        # Difference visualization
        diff_image = np.abs(test_image_reshaped - mean_normal_image)
        plt.subplot(1, 3, 2)
        plt.imshow(diff_image, cmap='hot')
        plt.title("Difference Map")
        
        # Heatmap of anomaly score
        anomaly_score = test_image.reshape(128, 128) - mean_normal_image
        plt.subplot(1, 3, 3)
        sns.heatmap(anomaly_score, cmap='coolwarm', center=0)
        plt.title("Anomaly Heatmap")
    
    plt.show()
else:
    print("Error loading test image.")
