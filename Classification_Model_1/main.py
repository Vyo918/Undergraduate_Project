import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from pathlib import Path
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(120)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 120)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# # data transforms for train and test datasets
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                             [0.229, 0.224, 0.225])
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# train_dir = Path("dataset/train")  # train folders
# test_dir = Path("dataset/test")  # test folders

# train_datasets = datasets.ImageFolder(root=train_dir, transform=train_transform)
# test_datasets = datasets.ImageFolder(root=test_dir, transform=test_transform)

# train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True, num_workers=4)
# test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False, num_workers=4)

# class_names = train_datasets.classes


# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to match LeNet input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Apply data augmentation for the training dataset
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Update the dataset loaders with augmented data
trainset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

model = LeNet5().to(device)
loss_fn = nn.CrossEntropyLoss()  # CrossEntropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower initial learning rate
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

# Apply the weight initialization to the model
model.apply(init_weights)





# model = MODEL_NAME(num_classes=len(class_names)).to(device)

# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# training
def train(model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = device
        ):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        output = model(images)
        loss = loss_fn(output, labels)
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
# Testing
def evaluate(model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device = device
        ):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
epochs = 20

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    train(model, train_dataloader, loss_fn, optimizer, device)
    evaluate(model, test_dataloader, loss_fn, device)
    scheduler.step()