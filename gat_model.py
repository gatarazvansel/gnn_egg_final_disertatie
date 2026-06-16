from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool, global_max_pool


class EEGGAT(nn.Module):
    """
    Graph Attention Network for EEG graph classification.
    3-layer GAT with residual connections, dual pooling, and MLP classifier.
    """

    def __init__(
        self,
        num_node_features: int = 4,
        hidden_channels: int = 64,
        num_classes: int = 2,
        heads_first_layer: int = 4,
        heads_second_layer: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.heads_first_layer = heads_first_layer
        self.heads_second_layer = heads_second_layer
        self.dropout = dropout

        self.input_norm = BatchNorm(num_node_features)

        # Layer 1: multi-head attention, output concatenated
        self.gat1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            heads=heads_first_layer,
            concat=True,
            dropout=dropout * 0.3,
        )
        self.bn1 = BatchNorm(hidden_channels * heads_first_layer)

        # Layer 2: reduce back to hidden_channels
        self.gat2 = GATConv(
            in_channels=hidden_channels * heads_first_layer,
            out_channels=hidden_channels,
            heads=heads_second_layer,
            concat=True,
            dropout=dropout * 0.3,
        )
        self.bn2 = BatchNorm(hidden_channels * heads_second_layer)

        # Layer 3: final graph conv
        self.gat3 = GATConv(
            in_channels=hidden_channels * heads_second_layer,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout * 0.3,
        )
        self.bn3 = BatchNorm(hidden_channels)

        # Residual projection (for skip connection from layer 1 output to layer 3)
        self.residual_proj = nn.Linear(
            hidden_channels * heads_first_layer,
            hidden_channels,
            bias=False,
        )

        # MLP Classifier (dual pooling: mean + max -> 2 * hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_norm(x)

        # Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        # Save for residual
        residual = self.residual_proj(x)

        # Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        # Layer 3 with residual from layer 1
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = x + residual  # residual skip connection

        # Dual global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP classifier
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    from torch_geometric.data import Data

    x = torch.rand((5, 4), dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4],
         [1, 0, 2, 1, 4, 3]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [[0.8], [0.8], [0.9], [0.9], [0.7], [0.7]],
        dtype=torch.float32,
    )
    batch = torch.zeros(5, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    model = EEGGAT(
        num_node_features=4,
        hidden_channels=32,
        num_classes=3,
        heads_first_layer=4,
        heads_second_layer=1,
        dropout=0.5,
    )

    out = model(graph.x, graph.edge_index, batch, graph.edge_attr)
    print("Output shape:", out.shape)
    print("Output:", out)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
