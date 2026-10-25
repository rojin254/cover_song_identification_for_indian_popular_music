from torch import nn

from .CQTNet import CQTNet


class ExperimentModel(nn.Module):
    """Explicit reconstructed dropout after fc0, never inside the backbone."""

    def __init__(self, classes, settings):
        super().__init__()
        self.backbone = CQTNet()
        if classes != 10000:
            self.backbone.fc1 = nn.Linear(300, classes)
        self.dropout = nn.Dropout(settings["dropout"])
        self.settings = settings
        if settings["freeze_conv0"]:
            self.backbone.features.conv0.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.settings["batchnorm"] == "frozen":
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x):
        if self.dropout.p == 0 or not self.training:
            return self.backbone(x)
        backbone = self.backbone
        features = backbone.pool(backbone.features(x)).flatten(1)
        embedding = self.dropout(backbone.fc0(features))
        return backbone.fc1(embedding), embedding
