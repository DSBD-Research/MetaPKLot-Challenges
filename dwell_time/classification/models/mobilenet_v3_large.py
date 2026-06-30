import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torchvision.models import MobileNet_V3_Large_Weights as pre_weights
from torchvision.models import mobilenet_v3_large as mobilenet
import importlib
import torch

NUM_CLASSES = 2

class model():
    def __init__(self, pre_trained=True, config=None):
        self.model = mobilenet(weights=pre_weights.IMAGENET1K_V2) if pre_trained else mobilenet()
        self.config = config

        self.model.classifier[-1] = torch.nn.Linear(1280, NUM_CLASSES)
        self.model = self.model.to(self.config["device"])

        for name, param in self.model.named_parameters():
            if "classifier" not in name and "features.16" not in name:
                param.requires_grad = False

        loss_module = importlib.import_module("torch.nn")
        self.loss = getattr(loss_module, self.config["loss"])()

        optimizer_module = importlib.import_module("torch.optim")
        self.optimizer = getattr(optimizer_module, self.config["optimizer"])(self.model.parameters(), lr=self.config["lr"])

    
    def get_state_dict(self):
        return self.model.state_dict()

    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.config["device"])


    def train(self, loader):
        self.model.train()

        device = self.config["device"]
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()

            preds = self.model(inputs)
            loss = self.loss(preds, labels)

            loss.backward()
            self.optimizer.step()

    
    def validate(self, loader):
        self.model.eval()

        device = self.config["device"]
        loss_module = importlib.import_module("torch.nn")
        local_loss = getattr(loss_module, self.config["loss"])()
        final_loss = 0.0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = self.model(inputs)

                loss = local_loss(preds, labels)
                final_loss += loss.item()
            
        return final_loss
    

    def predict(self, loader, threshold=0.5):
        self.model.eval()

        device = self.config["device"]
        final_preds, final_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                probs = torch.sigmoid(self.model(inputs))
                class_1_probs = probs[:, 1]

                predicted = (class_1_probs > threshold).long()
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
        
        return (final_preds, final_labels)
    

    def probabilities(self, loader):
        self.model.eval()

        device = self.config["device"]
        final_probs, final_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                probs = torch.sigmoid(self.model(inputs))

                final_probs.extend(probs.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
        
        return (final_probs, final_labels) 