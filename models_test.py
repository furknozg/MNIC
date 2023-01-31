
import data_classes as dconst
import mal_models as malmod
import torch
from torch.utils import data
import mal_models as malmod
from torch import nn
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import OneClassSVM
import numpy as np

from pipeline import MNIC

device = "cuda" if torch.cuda.is_available() else "cpu"
#Identification

PATH = "./Models/ember2018/identifier.pt"
test_dataset = dconst.EmberDataset(data_path='./Datasets/ember2018/', train = False)

feature_len = -1
for x,_ ,labels in test_dataset:
    feature_len = x.shape[0]
    break

identifier = malmod.MalwareIdentifier(input_size = feature_len, hidden_size = 600, output_size = 3)


identifier.load_state_dict(torch.load(PATH))

criterion = nn.NLLLoss()
optimizer = None

running_corrects = 0

batch_size = 1
data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("*"*50)
print("Testing Identifier")
for inputs, labels, _ in data_loader:
    # Move the inputs and labels to the device where the model is located
    inputs = inputs
    labels = labels

    # Set the model to eval mode
    identifier.eval()

    # Forward pass
    logits = identifier(inputs)
    _, preds = torch.max(logits, 1)

    # Update the running corrects count
    if labels.data == preds:
        running_corrects += 1

    
accuracy = running_corrects / len(test_dataset)
print('Test accuracy: ', accuracy)


# Classification

PATH = "./Models/ember2018/classifier_restricted.pt"

test_dataset_malonly = dconst.EmberDataset(data_path='./Datasets/ember2018/', train = False, malonly = True, classifier = True)

class_len = test_dataset_malonly.class_length()


classifier = malmod.MalwareClassifier(input_size = feature_len, hidden_size = 800, num_classes = class_len)

classifier.load_state_dict(torch.load(PATH))

criterion = nn.NLLLoss()
optimizer = None

running_corrects = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

data_loader = data.DataLoader(test_dataset_malonly, batch_size=batch_size, shuffle=True)

print("*"*50)
print("Testing Classifier")
for inputs, _, labels in data_loader:
    # Move the inputs and labels to the device where the classifier is located
    inputs = torch.from_numpy(np.array(inputs))
    labels = labels

    # Set the model to eval mode
    classifier.eval()

    # Forward pass
    logits = classifier(inputs)
    _, preds = torch.max(logits, 1)

    # Update the running corrects count
    if labels.data == preds:
        running_corrects += 1

    
accuracy = running_corrects / len(test_dataset_malonly)
print('Test accuracy: ', accuracy)

PATH = "./Models/ember2018/iforest.pickle"

iso_forest = pickle.load(open(PATH, "rb"))


print("*"*50)
print("Testing Isolation Forest")
running_corrects = 0

for inputs, labels, classes in test_dataset:

    preds = iso_forest.predict(inputs.reshape(1,-1))

    if test_dataset.de_enumerate_idx(classes.data) != "Benign" and preds == 1:
            running_corrects += 1
    elif test_dataset.de_enumerate_idx(classes.data) == "Benign" and preds == -1:
            running_corrects += 1



accuracy = running_corrects / len(test_dataset)
print('Test accuracy: ', accuracy)



print("*"*50)
print("Testing Pipeline")

# Assumption taken in pipeline (Real world applications will stumble on more benign data than malware)
mnic_dataset = dconst.EmberDataset(data_path='./Datasets/ember2018/', train = False)

mnic_pipeline = MNIC(identifier, classifier, iso_forest, mnic_dataset, test_dataset_malonly)


mnic_loader = data.DataLoader(mnic_dataset, batch_size=batch_size, shuffle=True)

running_corrects = 0

for input, label, class_idx in mnic_loader:
    pred = mnic_pipeline.pipeline_eval(input)
    

    if test_dataset.de_enumerate_idx(class_idx.data[0]) == pred: # Cross examination of the label string to the predicted string
        running_corrects += 1

accuracy = running_corrects / len(test_dataset)
print('Test accuracy: ', accuracy)

