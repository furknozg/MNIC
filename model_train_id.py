import data_classes as dconst
import mal_models as malmod
import torch
from torch.utils import data
import mal_models as malmod
from torch import nn
import matplotlib.pyplot as plt


# Test code for loading ember data
# Identification
PATH = "./Models/ember2018/identifier.pt"
print("*"*50)
print("Training Identifier")

dataset = dconst.EmberDataset(data_path='./Datasets/ember2018/', train = True,malonly=False)

batch_size = 128
data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Pull in shape from the first sample
feature_len = -1
for dp, labels, _ in data_loader:
    feature_len = dp.shape[1]
    break
# Create an instance of the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = malmod.MalwareIdentifier(input_size = feature_len, hidden_size = 600, output_size = 3).to(device)


# Define the loss function and the optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum= 0.9, nesterov = True)


# Train the model (Identification)
print(f"Running Identifier on: {dataset} with batch-size:{batch_size}, Using-device: {device}")
losses = []
for epoch in range(3500):  # Run for 1000 epochs            
    running_loss = 0.0

    for X_batch, y_batch, _ in data_loader:  # Iterate over the batches
        # mediate data derivation and add to device (cuda or cpu)
        y_batch = torch.add(y_batch.type(torch.LongTensor),1)
        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()
        
        
        running_loss += loss.item() * X_batch.size(0) 
    epoch_loss = running_loss / len(data_loader)
    losses.append(epoch_loss)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | loss: {epoch_loss}")


torch.save(model.state_dict(), PATH)

plt.plot([*range(1, 3501, 1)], losses)
plt.savefig("./pics/identifier_epochs.png")


