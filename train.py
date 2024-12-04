import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import ConvNet

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset with augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),  # Add slight rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add slight shift
    ])

    train_dataset = datasets.MNIST(root='./data', 
                                 train=True,
                                 transform=transform,
                                 download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=64,  # Smaller batch size
                                             shuffle=True)

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Train the model
    total_step = len(train_loader)
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f'Step [{i+1}/{total_step}], Loss: {avg_loss:.4f}')
            scheduler.step(avg_loss)
            running_loss = 0.0

    # Save the model
    saved_path = model.save_model()
    print(f"Model saved to {saved_path}")
    return model

if __name__ == "__main__":
    train() 