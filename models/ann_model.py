import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc_emotion = nn.Linear(hidden_dim, 1)
        self.fc_polarity = nn.Linear(hidden_dim, num_classes)
        self.fc_empathy = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        return self.fc_emotion(h), self.fc_polarity(h), self.fc_empathy(h)
