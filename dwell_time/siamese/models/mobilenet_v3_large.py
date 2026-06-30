import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torchvision.models import MobileNet_V3_Large_Weights as pre_weights
from torchvision.models import mobilenet_v3_large as mobilenet
import torch.nn.functional as func
import importlib
import torch

class model(torch.nn.Module):
    def __init__(self, pre_trained=True, config=None):
        super(model, self).__init__()

        backbone = mobilenet(weights=pre_weights.IMAGENET1K_V2) if pre_trained else mobilenet()
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.fc = torch.nn.Linear(960, 128)
        self.l2norm = func.normalize

        self.config = config
        self.loss = torch.nn.TripletMarginLoss(swap=True, margin=0.2)

        optimizer_module = importlib.import_module("torch.optim")
        self.optimizer = getattr(optimizer_module, self.config["optimizer"])(self.parameters(), lr=self.config["lr"])

        self.to(device=config["device"])


    def forward(self, inputs):
        towers = [self.features(inp) for inp in inputs]
        towers = [torch.flatten(tower, 1) for tower in towers]
        towers = [self.fc(tower) for tower in towers]
        towers = [self.l2norm(tower, dim=1) for tower in towers]

        return towers


    def train_model(self, loader):
        self.train()

        device = self.config["device"]
        final_loss = 0.0

        for inputs in loader:
            inputs = [inp.to(device) for inp in inputs]
            self.optimizer.zero_grad()
            
            a_pred, p_pred, n_pred = self(inputs)
            loss = self.loss(a_pred, p_pred, n_pred)
            final_loss += loss.item()

            loss.backward()
            self.optimizer.step()
        
        return final_loss


    def validate(self, loader):
        self.eval()

        device = self.config["device"]
        local_loss = torch.nn.TripletMarginLoss(swap=True, margin=0.2)
        final_loss = 0.0

        with torch.no_grad():
            for inputs in loader:
                inputs = [inp.to(device) for inp in inputs]
                a_pred, p_pred, n_pred = self(inputs)

                loss = local_loss(a_pred, p_pred, n_pred)
                final_loss += loss.item()
            
        return final_loss
    

    def distances(self, loader):
        self.eval()

        device = self.config["device"]
        final_dists, final_labels = [], []

        with torch.no_grad():
            for inputs in loader:
                inputs = [inp.to(device) for inp in inputs]
                a_pred, p_pred, n_pred = self(inputs)

                pos_dist = func.pairwise_distance(a_pred, p_pred)
                neg_dist = func.pairwise_distance(a_pred, n_pred)

                all_dists = torch.cat([pos_dist, neg_dist], dim=0)
                labels = torch.cat([torch.zeros(len(pos_dist)), torch.ones(len(neg_dist))], dim=0)

                final_dists.extend(all_dists.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
        
        return (final_dists, final_labels)


    def predict(self, loader, threshold=0.5):
        self.eval()

        device = self.config["device"]
        final_preds, final_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = [inp.to(device) for inp in inputs]

                tensors = self(inputs)
                dists = func.pairwise_distance(tensors[0], tensors[1])

                predicted = (dists > threshold).long()
                
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
        
        return (final_preds, final_labels)
    

    def predict_distances(self, loader):
        self.eval()

        device = self.config["device"]
        final_distances = []
        final_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = [inp.to(device) for inp in inputs]

                tensors = self(inputs)
                distances = func.pairwise_distance(tensors[0], tensors[1])

                final_distances.extend(distances.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

        return final_distances, final_labels