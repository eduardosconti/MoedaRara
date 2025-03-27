import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os

# Pré-processamento das imagens
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# Diretório dos dados
data_dir = "C:/Users/luiza/MoedaRara/modelos/Cruzeiro_Novo"
dataset = ImageFolder(root=data_dir, transform=transform)

# Divisão entre treino e teste
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Definição da CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 5)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializa modelo, loss e otimizador
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Verifica se já existe um modelo treinado salvo
modelo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_treinado.pth")

if os.path.exists(modelo_path):
    print("Carregando modelo já treinado...")
    model.load_state_dict(torch.load(modelo_path))
    model.eval()  # Coloca em modo de avaliação para uso imediato
    print("Modelo carregado com sucesso!")
else:
    # Treinamento do modelo
    print("Treinando o modelo com Data Augmentation ajustado...")
    for epoch in range(15):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Época {epoch+1}, Perda: {total_loss / len(train_loader):.4f}')

    print("Treinamento concluído!")

    # Salva o modelo treinado para evitar novo treinamento no futuro
    torch.save(model.state_dict(), modelo_path)
    print("Modelo salvo com sucesso!")

# Avaliação do modelo
print("Avaliando o modelo...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Precisão do modelo: {accuracy:.2f}%')

# Exemplo de previsões
print("\nExemplo de previsões:")
classes = dataset.classes
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(images)):
            print(f"Imagem {i+1}: Real = {classes[labels[i]]}, Previsto = {classes[predicted[i]]}")
        break  


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, random_split
# import torch.nn.functional as F



# transform = transforms.Compose([
#     transforms.Grayscale(),  
#     transforms.Resize((128, 128)),  
#     transforms.RandomRotation(10),  
#     transforms.RandomHorizontalFlip(), 
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
#     transforms.ToTensor(), 
# ])


# data_dir = "C:/Users/luiza/MoedaRara/modelos/Cruzeiro_Novo"
# dataset = ImageFolder(root=data_dir, transform=transform)


# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Aumento do batch_size para maior estabilidade
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # unique_labels = set()
# # for _, labels in train_loader:
# #     unique_labels.update(labels.numpy())
# # print("Classes presentes no dataset:", unique_labels)




# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 32 * 32, 128)  
#         self.fc2 = nn.Linear(128, 4)  

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))  
#         x = x.view(-1, 64 * 32 * 32)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x



# model = CNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# print("Treinando o modelo com Data Augmentation ajustado...")
# for epoch in range(15):
#     model.train()
#     total_loss = 0
#     for images, labels in train_loader:
#         print("Labels recebidos:", labels) ######
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Época {epoch+1}, Perda: {total_loss / len(train_loader):.4f}')

# print("Treinamento concluído!\n")



# print("Avaliando o modelo...")
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()


# accuracy = 100 * correct / total
# print(f'Precisão do modelo: {accuracy:.2f}%')



# print("\nExemplo de previsões:")
# classes = dataset.classes
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         for i in range(len(images)):
#             print(f"Imagem {i+1}: Real = {classes[labels[i]]}, Previsto = {classes[predicted[i]]}")
#         break  
