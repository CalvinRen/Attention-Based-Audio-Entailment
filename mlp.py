import torch.nn as nn

class AudioTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AudioTextClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)