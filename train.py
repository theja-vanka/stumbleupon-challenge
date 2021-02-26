import torch
import torch.nn as nn
from model.data_loader import Data_Loaders
from model.net import StumbleNet


def train_model(num_epochs):
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 350
    itrloss = 100

    data_loaders = Data_Loaders(batch_size)
    model = StumbleNet(data_loaders.features, 256, 3, 0.2, device)
    model.to(device)

    criterion = nn.BCELoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for idx, (inputs, labels) in enumerate(data_loaders.train_loader):
            # Reshaping and assigning to device
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            labels = labels.float()
            labels = labels.to(device)
            # Model run
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Reset and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f'Step: {idx+1}, loss: {loss.item():.4f}')
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

        with torch.no_grad():
            validationloss = []
            for idx, (inputs, labels) in enumerate(data_loaders.test_loader):
                inputs = inputs.unsqueeze(0)
                inputs = inputs.to(device)
                labels = labels.float()
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validationloss.append(loss.item())
            loss = sum(validationloss)/len(validationloss)
            if loss < itrloss:
                itrloss = loss
                PATH = f"./experiments/weights/weights_{loss*10**8:.3f}.pkl"
                torch.save(model, PATH)


if __name__ == '__main__':
    no_epochs = 300
    train_model(no_epochs)
