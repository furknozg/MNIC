import data_classes as dconst
import mal_models as malmod
import torch
from torch.utils import data
import mal_models as malmod
from torch import nn
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import ember
import pickle

# Import model
from sklearn.ensemble import IsolationForest



class MNIC:# Malware Neural Identifier and Classifier

    def __init__(self, identifier: malmod.MalwareIdentifier, classifier: malmod.MalwareClassifier, iso_forest: IsolationForest, dset: dconst.EmberDataset, dset_mal: dconst.EmberDataset) -> None:
        self.identifier = identifier
        self.classifier = classifier
        self.iso_forest = iso_forest
        self.dset = dset
        self.dset_mal = dset_mal

    def pipeline_eval(self,data_point):
        data_point = data_point
        self.identifier.eval()
        self.classifier.eval()
        

        logits_id = self.identifier(data_point)
        _, preds_id = torch.max(logits_id, 1)

        if preds_id == 0:
            # Instant benign
            return "Benign"
        elif preds_id == 1:
            # Instant malware
            logits_fam = self.classifier(torch.from_numpy(np.array(data_point)))
            _, preds_family = torch.max(logits_fam, 1)
            
            return self.dset_mal.de_enumerate_idx(preds_family.data[0])
        elif preds_id == -1:
            # Suspicious conditions / Indecisive
            preds_anomaly = self.iso_forest.predict(data_point.reshape(1,-1))
            if preds_anomaly == 1:
                logits_fam = self.classifier(torch.from_numpy(np.array(data_point)))
                _, preds_family = torch.max(logits_fam, 1)
                return self.dset_mal.de_enumerate_idx(preds_family.data[0])

            else:
                return "Benign"


