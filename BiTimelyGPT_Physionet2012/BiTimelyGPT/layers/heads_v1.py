import torch.nn as nn
import torch.nn.functional as F
from layers.snippets import SigmoidRange

class PhysioNetHead(nn.Module):
    """Head for PhysioNet mortality prediction and risk estimation"""
    def __init__(self, d_model):
        super(PhysioNetHead, self).__init__()
        self.dense = nn.Linear(d_model, 2)  # Two outputs: mortality and risk

    def forward(self, x):
        """
        x: tensor [batch_size x d_model]
        output: tensor [batch_size x 2] (mortality logit, risk logit)
        """
        outputs = self.dense(x)
        # First output is mortality logit, second is risk estimation
        mortality_logit = outputs[:, 0]
        risk_logit = F.sigmoid(outputs[:, 1])  # Ensure risk is between 0 and 1
        return torch.stack([mortality_logit, risk_logit], dim=1)

# Keep existing head classes unchanged
class PretrainHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(PretrainHead, self).__init__()
        self.head = nn.Linear(d_model, n_output)

    def forward(self, x):
        return self.head(x)

class ClfHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(ClfHead, self).__init__()
        self.head = nn.Linear(d_model, n_output)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.head(x)

class RegrHead(nn.Module):
    def __init__(self, d_model, output_dim, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.regr_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.mean(dim=1)
        y = self.regr_layer(x)
        if self.y_range: 
            y = SigmoidRange(*self.y_range)(y)
        return y