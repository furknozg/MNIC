# Ember Dataset
import torch
import torch.utils.data as data
import ember
import pandas as pd
import numpy as np
# Ember Dataset
class EmberDataset(data.Dataset):
    '''A dataset torch encapsulator for the ember malware classifier data.

        Example usage:
        dataset = EmberDataset(data_path='Datasets/ember2018/', train = False)

        batch_size = 64
        
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)'''
        
    def __init__(self, data_path, transform=None, train= None, malonly = None, classifier = None):
        # Load the data from the given path
        X_train, y_train, X_test, y_test = [None, None, None, None]
        try:
            X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_path)

        except:
            print("The dataset folder is not vectorized, this might take a while...")
            self.create_dat_file(data_path)
            X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_path)
        
        if malonly:
                print("DISCLAIMER: Extracting malware only data, which might take a minute...")

                X_train = X_train[y_train == 1]
                y_train = y_train[y_train == 1]
                X_test = X_test[y_test == 1]
                y_test = y_test[y_test == 1]

        
        self.class_labels = ember.read_metadata(data_path)
        
        if malonly:
            self.class_labels = self.class_labels[self.class_labels["label"] == 1].reset_index()
        
        self.class_labels = self.class_labels["avclass"].fillna("benign")
        self.class_label_idx = pd.Categorical(self.class_labels).codes - 1
        if classifier:
            # Re evaluate the arrays (Note this compartment uses 70 30 distribution for train-test)
            select_df_of_mals = self.class_labels[(self.class_labels == "benign") | (self.class_labels == "sivis") | (self.class_labels == "emotet") | (self.class_labels == "flystudio") | (self.class_labels == "zbot") | (self.class_labels == "xtrat") | (self.class_labels == "ursnif") | (self.class_labels == "installmonster") | (self.class_labels == "startsurf") | (self.class_labels == "dinwod") | (self.class_labels == "upatre")& (self.class_labels == "sality") | (self.class_labels == "wannacry") | (self.class_labels == "high") | (self.class_labels == "zusy") | (self.class_labels == "chapak") | (self.class_labels == "downloadguide") | (self.class_labels == "gamehack") | (self.class_labels == "virlock")].dropna()
            self.class_labels = select_df_of_mals
            idx = self.class_labels.index.values.tolist()
            
            #print(np.take(np.vstack((X_train, X_test)),idx, axis = 0).shape)


            X_definitive = np.take(np.vstack((X_train, X_test)),idx, axis = 0)
            Y_definitive = np.take(np.append(y_train, y_test), idx)
            
            #print(X_definitive.shape)
            #print(X_definitive[81339])
            
            self.class_labels = self.class_labels.reset_index()["avclass"]
            
            #print(self.class_labels)
            
            self.class_label_idx = pd.Categorical(self.class_labels).codes - 1
            if train:
                if malonly:
                    self.data = X_definitive[0:117267]
                    self.label = Y_definitive[0:117267]
                    self.class_data = self.class_label_idx[0:117267]
                else:
                    self.data = X_definitive[0:493203]
                    self.label = Y_definitive[0:493203]
                    self.class_data = self.class_label_idx[0:493203]
            elif not train:
                if malonly:
                    self.data = X_definitive[117267:167525]
                    self.label = Y_definitive[117267:167525]
                    self.class_data = self.class_label_idx[117267:167525]
                else:
                    self.data = X_definitive[493203:704676]
                    self.label = Y_definitive[493203:704676]
                    self.class_data = self.class_label_idx[493203:704576]
        else:
            # Legacy version, (still run on identifier and iso forest)
            if train:
                self.label = y_train
                self.data = X_train
                if malonly:
                    self.class_data = self.class_label_idx[0:300000]
                else:
                    self.class_data = self.class_label_idx[0:800000]
            elif not train:
                self.label = y_test
                self.data = X_test
                if malonly:
                    self.class_data = self.class_label_idx[300000:400000]
                else:
                    self.class_data = self.class_label_idx[800000:1000000]
        
        self.transform = transform
        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)
    
    def __getitem__(self, index):
        #print(f"data: {self.data} label: {self.class_data} class: {self.label}")
        data = self.data[index]
        label = self.label[index]
        class_data = self.class_data[index]
            

        if self.transform:
            # Apply any transformation to the data
            data = self.transform(data)
        
        return data, label, class_data
    
    def create_dat_file(data_path):
        # Do this once for every machine, after the data is vectorized the dat file can stay
        ember.create_vectorized_features(data_path)
        ember.create_metadata(data_path)
    def de_enumerate_idx(self, idx):
        return pd.Categorical(self.class_labels)[idx]
    def class_length(self):
        return self.class_labels.nunique()
