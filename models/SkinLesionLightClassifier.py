import timm
import torch.nn as nn


class SkinLesionLightClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name="mobilenetv3_small_050", pretrained=True)

        num_in_features = self.model.get_classifier().in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=1024, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Unflatten(1, (1, 32, 32)),
            nn.Conv2d(1, 25, kernel_size=5, stride=1, padding=0), # before 25
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        return self.model(X)

