
import data_classes as dconst
import mal_models as malmod
import torch
from torch.utils import data
import mal_models as malmod
from torch import nn
import matplotlib.pyplot as plt
# Classification
PATH = "./Models/ember2018/classifier_restricted.pt"
print("*"*50)
print("Training Classifier")

dataset_malonly = dconst.EmberDataset(data_path='./Datasets/ember2018/', train = True, malonly = True, classifier=True)
# This takes a while

class_len = dataset_malonly.class_length()


batch_size = 128
data_loader = data.DataLoader(dataset_malonly, batch_size=batch_size, shuffle=True)


feature_len = -1
for dp,_ ,labels in data_loader:
    feature_len = dp.shape[1]
    break


# Create an instance of the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = malmod.MalwareClassifier(input_size = feature_len, hidden_size = 800, num_classes = class_len).to(device)

# Define the loss function and the optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, nesterov=True)


# Train the model (Classification)
print(f"Running Classifier on: {dataset_malonly} with batch-size:{batch_size}, Using-device: {device}")
losses = []
for epoch in range(3500):  # Run for 1000 epochs            
    running_loss = 0.0

    for X_batch, _, y_batch in data_loader:  # Iterate over the batches
        
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
plt.plot([*range(1, 5001, 1)], losses)
plt.savefig("./pics/classifier_epochs.png")


