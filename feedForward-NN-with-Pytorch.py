import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from nnClasse import NeuralNet

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784 #28x28
num_classes = 10
hidden_size = 500
num_epochs = 10
learning_rate = 0.001
batch_size = 100

# using MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True,
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False,
                                          transform=transforms.ToTensor()  )

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,                                
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size,       
                                          shuffle=False)

# check one batch of the datapython

sampleData = iter(train_loader)
sample, labels = next(sampleData)
#print(sample.shape, labels.shape)

# plot few sample

for i in range(2):
    plt.subplot(2, 2, i+1)
    plt.imshow(sample[i][0], cmap='gray')
#plt.show()

    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_num_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{total_num_steps}, Runing_Loss: {loss.item():.4f}')
            
# Test the model
with torch.no_grad():
    num_correct = 0.0
    num_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # max returns (value, index)
        num_samples += labels.size(0)
        num_correct += (predicted == labels).sum().item()
    
    
        accuracy = 100.0*num_correct/num_samples
    
    print(f'\nTest accuracy of the network on the 10,000 test images: {accuracy} %')
    
    print('\nTraining completed')
    



