import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. HYPERPARAMETERS (Dễ dàng thay đổi để thử nghiệm)
# ---------------------------------------------------------
batch_size = 64
learning_rate = 1e-2
epochs = 10

# Danh sách 10 lớp nhãn của FashionMNIST
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
# ---------------------------------------------------------
# 2. LOAD DATASET & TRANSFORMS
# ---------------------------------------------------------
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------
# 3. BUILD NEURAL NETWORK
# ---------------------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------
# 4. TRAINING & EVALUATION LOOPS
# ---------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, accuracy

# Chạy huấn luyện và lưu lịch sử Loss
train_losses = []
test_losses = []

print("--- Starting Training ---")
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}")
    t_loss = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = test(test_dataloader, model, loss_fn)
    train_losses.append(t_loss)
    test_losses.append(val_loss)

print("Done Training!\n")

# ---------------------------------------------------------
# 5. SAVE AND LOAD MODEL
# ---------------------------------------------------------
# Save
model_path = "model.pth"
torch.save(model.state_dict(), model_path)
print(f"[1] Saved PyTorch Model State to {model_path}")

# Load
loaded_model = NeuralNetwork().to(device)
loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
loaded_model.eval()
print("[2] Successfully Loaded PyTorch Model State\n")

# ---------------------------------------------------------
# 6. TASK: VISUALIZE LOSS GRAPH
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Testing Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
print("[3] Loss graph saved as 'loss_graph.png'")
plt.show()

# ---------------------------------------------------------
# 7. TASK: DISPLAY PREDICTED VS ACTUAL IMAGES
# ---------------------------------------------------------
figure = plt.figure(figsize=(10, 8))
cols, rows = 3, 3

# Lấy 1 batch dữ liệu test để hiển thị
X_test, y_test = next(iter(test_dataloader))
X_test_dev = X_test.to(device)

with torch.no_grad():
    predictions = loaded_model(X_test_dev)

for i in range(1, cols * rows + 1):
    img = X_test[i-1].squeeze().numpy()
    actual_label = classes[y_test[i-1].item()]
    pred_label = classes[predictions[i-1].argmax(0).item()]
    
    # Màu xanh nếu đoán đúng, màu đỏ nếu đoán sai
    color = "green" if actual_label == pred_label else "red"
    
    figure.add_subplot(rows, cols, i)
    plt.title(f"Pred: {pred_label}\nActual: {actual_label}", color=color, fontsize=10)
    plt.axis("off")
    plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.savefig("predictions.png")
print("[4] Predictions visualization saved as 'predictions.png'")
plt.show()