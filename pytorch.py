# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# import numpy as np
# from cattorch import transpile, verify

# class Binarize(object):
#     def __call__(self, img):
#         return (img > 0.5).float()

# # MNIST の読み込み（28x28、白=1 黒=0）
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     Binarize()
# ])

# train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# # モデル構成
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 255)
#         self.fc2 = nn.Linear(255, 200)
#         self.fc3 = nn.Linear(200, 150)
#         self.fc4 = nn.Linear(150, 10)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# model = MLP()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 学習率を自動で下げる
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# # 学習
# epochs = 15
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0
#     correct = 0
#     total = 0

#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     epoch_loss = running_loss / len(train_loader)
#     epoch_acc = correct / total

#     print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.4f}")

#     scheduler.step()

# # # CSV 出力
# # def save_layer(layer, name):
# #     W = layer.weight.detach().cpu().numpy()
# #     b = layer.bias.detach().cpu().numpy()

# #     np.savetxt(f"{name}_W.csv", W.reshape(-1), delimiter=",")
# #     np.savetxt(f"{name}_b.csv", b, delimiter=",")

# # save_layer(model.fc1, "weight_1")
# # save_layer(model.fc2, "weight_2")
# # save_layer(model.fc3, "weight_3")
# # save_layer(model.fc4, "weight_4")

# # print("CSV 出力完了")


# model.eval()

# # Scratch に渡すためのダミー入力（MNIST 28x28）
# example_input = torch.randn(1, 1, 28, 28)

# # Scratch 用スプライト生成
# artifact = transpile(model, example_input, "build/mnist_mlp")
# print("生成された Scratch スプライト:", artifact.path)

# # Scratch 内で正しく動くか検証
# verify(model, example_input, artifact)
# print("Scratch 変換の検証が完了しました")

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

# 二値化フィルタ
class Binarize(object):
    def __call__(self, img):
        return (img > 0.5).float()

# MNIST 読み込み（あなたの設定と同じ）
transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize()
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# ギャラリー表示（最初の 25 枚）
plt.figure(figsize=(10, 10))

for i in range(25):
    img, label = dataset[i]
    img = img.squeeze(0).numpy()  # 28x28 に変換

    plt.subplot(5, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
