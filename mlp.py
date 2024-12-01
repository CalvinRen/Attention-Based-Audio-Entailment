import torch.nn as nn

# Define the model
class AudioTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super(AudioTextClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, sample_embeddings):
        logits = self.classifier(sample_embeddings)
        return logits