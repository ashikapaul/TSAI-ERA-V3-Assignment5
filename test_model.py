import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import ConvNet
import pytest
import glob
import os

def test_model_architecture():
    model = ConvNet()
    
    # Test 1: Check model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"
    
    # Test 2: Check input shape compatibility
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    # Check if model exists
    model_files = glob.glob("model_*.pth")
    if not model_files:
        pytest.skip("No trained model found. Please run 'python train.py' first")
    
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data',
                                train=False,
                                transform=transform,
                                download=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1000,
                                            shuffle=False)
    
    # Load the model
    model = ConvNet()
    latest_model = max(model_files, key=lambda x: x.split('_')[2])
    model.load_state_dict(torch.load(latest_model))
    model.eval()
    
    # Test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%"

if __name__ == "__main__":
    test_model_architecture()
    test_model_accuracy() 