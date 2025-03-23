import os
import cv2
import time
import numpy as np
from PIL import Image

debug = False


sourceBasePath = os.path.dirname(os.path.abspath(__file__)) + '\\Cruzeiro_Novo\\'

subfolders = ['1', '2', '5', '10', '20', '50', 'verso']  # List of subfolder names


# Limpa a ultima run (apaga tudo das pastas de output)
for folder in ['output treino', 'output teste']:
    for subfolder in subfolders:
        outputFolderPath = os.path.join(sourceBasePath, folder, subfolder + '\\')
        for item in os.listdir(outputFolderPath):
            os.remove(os.path.join(outputFolderPath, item))

processing_times = []

def preProcess(image_path, output_path):
    imgName = os.path.basename(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    edges = cv2.Canny(enhanced, 50, 150)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=30, maxRadius=0
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
    else:
        height, width = image.shape
        x, y = width // 2, height // 2
        r = min(width, height) // 2

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)

    if debug:
        cv2.imwrite(output_path + imgName + '_1_enhanced.png', enhanced)
        cv2.imwrite(output_path + imgName + '_2_edges.png', edges)
        cv2.imwrite(output_path + imgName + '_3_mask.png', mask)

    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    cv2.imwrite(output_path + '\\' + imgName, result)



total_start = time.time()

for folder in ['treino', 'teste']:
    for subfolder in subfolders:
        sourceFolderPath = os.path.join(sourceBasePath, folder, subfolder)
        destFolderPath = os.path.join(sourceBasePath, f"output {folder}", subfolder)

        # Create the destination folder if it doesn't exist
        os.makedirs(destFolderPath, exist_ok=True)
        
        for item in os.listdir(sourceFolderPath):
            image_path = os.path.join(sourceFolderPath, item)
            preProcess(image_path, destFolderPath)

total_time = time.time() - total_start  # Calculate total execution time
print(f"Total execution time: {total_time:.2f} seconds")

if processing_times:
    print(f"Max time: {max(processing_times):.2f} seconds")
    print(f"Min time: {min(processing_times):.2f} seconds")
