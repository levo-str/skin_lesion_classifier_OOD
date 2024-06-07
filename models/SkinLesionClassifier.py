import timm
import torch.nn as nn


class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name="resnet50", pretrained=True)

        num_in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=self.num_classes, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        return self.model(X)
