import os
import cv2
import numpy as np
from pathlib import Path

def generate_mean_image(coin_type, input_folder, output_folder):
    try:
        # Verifica se pasta de entrada existe
        if not os.path.exists(input_folder):
            print(f"ERRO: Pasta de entrada não existe: {input_folder}")
            return False

        # Cria pasta de saída se não existir
        os.makedirs(output_folder, exist_ok=True)

        # Lista arquivos válidos
        image_files = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"AVISO: Nenhuma imagem encontrada em {input_folder}")
            return False

        print(f"Processando {len(image_files)} imagens para {coin_type}...")

        mean_image = None
        processed_count = 0

        for filename in image_files:
            try:
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"AVISO: Não pude ler {filename}")
                    continue
                
                img = cv2.resize(img, (256, 256))
                img = img.astype(np.float32)
                
                if mean_image is None:
                    mean_image = img
                else:
                    mean_image += img
                
                processed_count += 1
            except Exception as e:
                print(f"ERRO processando {filename}: {str(e)}")

        if processed_count == 0:
            print("ERRO: Nenhuma imagem válida foi processada")
            return False

        mean_image /= processed_count
        mean_image = mean_image.astype(np.uint8)
        
        output_path = os.path.join(output_folder, f"{coin_type}_mean.png")
        success = cv2.imwrite(output_path, mean_image)
        
        if success:
            print(f"SUCESSO: Mean image salva em {output_path}")
            return True
        else:
            print("ERRO: Falha ao salvar a imagem")
            return False

    except Exception as e:
        print(f"ERRO CRÍTICO: {str(e)}")
        return False

if __name__ == "__main__":
    # Configurações - AJUSTE ESTES CAMINHOS!
    base_dir = r"C:\Users\luiza\MoedaRara\back\modelo\Cruzeiro_Novo"
    
    print("="*50)
    print("Iniciando geração de mean images")
    print("="*50)
    
    # Verifica estrutura de pastas
    required_folders = [
        os.path.join(base_dir, "output treino", "10"),
        os.path.join(base_dir, "output treino", "50"),
        os.path.join(base_dir, "output resultados")
    ]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"AVISO: Pasta não encontrada - {folder}")
    
    # Processa moedas
    results = {
        "10": generate_mean_image(
            coin_type="10",
            input_folder=os.path.join(base_dir, "output treino", "10"),
            output_folder=os.path.join(base_dir, "output resultados", "mean_images")
        ),
        "50": generate_mean_image(
            coin_type="50",
            input_folder=os.path.join(base_dir, "output treino", "50"),
            output_folder=os.path.join(base_dir, "output resultados", "mean_images")
        )
    }
    
    print("\nResumo:")
    for coin_type, success in results.items():
        status = "SUCESSO" if success else "FALHA"
        print(f"{coin_type} centavos: {status}")