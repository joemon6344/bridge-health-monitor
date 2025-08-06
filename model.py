import torch
import torch.nn as nn

class MetaConvLSTM(nn.Module):
    def __init__(self, sensor_input_dim, metadata_input_dim, hidden_dim, fusion_dim, num_classes):
        super(MetaConvLSTM, self).__init__()
        self.lstm = nn.LSTM(sensor_input_dim, hidden_dim, batch_first=True)
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_input_dim, fusion_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_sensor, x_meta):
        _, (h_n, _) = self.lstm(x_sensor)
        h_n = h_n[-1]
        meta_embed = self.meta_mlp(x_meta)
        fused = torch.cat([h_n, meta_embed], dim=1)
        return self.classifier(fused)
