import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from cattorch import transpile, verify

# 画像を2値化する前処理
class Binarize(object):
    def __call__(self, img):
        return (img > 0.5).float()

transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize()
])

# ここで自動ダウンロードが行われます（certifiのおかげでSSLエラーが防げます）
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 255)
        self.fc2 = nn.Linear(255, 200)
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.4f}")

    scheduler.step()

model.eval()

example_input = torch.randn(1, 1, 28, 28)

artifact = transpile(model, example_input, "build/mnist_mlp")
print("生成された Scratch スプライト:", artifact.path)

verify(model, example_input, artifact)
print("Scratch 変換の検証が完了しました")
