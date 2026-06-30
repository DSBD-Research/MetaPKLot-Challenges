import numpy as np
import torch

class ensemble_model():
    def __init__(self, module, state_dicts, config):
        self.models = []

        for state_dict in state_dicts:
            model = module.model(pre_trained=False, config=config)
            model.load_state_dict(state_dict)
            self.models.append(model)
        

    def predict(self, loader, threshold=0.5):
        models_outputs = []

        for model in self.models:
            models_outputs.append(model.probabilities(loader))
        
        probs = [output[0] for output in models_outputs]
        stacked_probs = np.array(probs)
        final_probs = np.mean(stacked_probs, axis=0).astype(np.float32)
        class_1_probs = final_probs[:, 1]
        
        preds = (class_1_probs > threshold).astype(int)
        labels = models_outputs[0][1]

        return (preds, labels)
