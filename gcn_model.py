from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, global_max_pool


class EEGGCN(nn.Module):
    """
    Graph Convolutional Network for EEG graph classification.
    3-layer GCN with ELU activation, uniform dropout, residual connections,
    dual pooling, and MLP classifier.
    """

    def __init__(
        self,
        num_node_features: int = 4,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.input_norm = BatchNorm(num_node_features)

        # Layer 1
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        # Layer 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        # Layer 3
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

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

    @staticmethod
    def _prepare_edge_weight(edge_attr: torch.Tensor | None) -> torch.Tensor | None:
        """
        Converts edge_attr to edge_weight format expected by GCNConv.
        """
        if edge_attr is None:
            return None

        if edge_attr.dim() == 1:
            return edge_attr

        if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
            return edge_attr.squeeze(1)

        raise ValueError(
            f"Unsupported edge_attr shape for GCN edge weights: {tuple(edge_attr.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_weight = self._prepare_edge_weight(edge_attr)

        x = self.input_norm(x)

        # Layer 1
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout * 0.4, training=self.training)

        # Layer 2 with residual
        identity = x
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout * 0.4, training=self.training)
        x = x + identity  # residual connection

        # Layer 3 with residual
        identity = x
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout * 0.4, training=self.training)
        x = x + identity  # residual connection

        # Dual global pooling (captures both average and salient features)
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

    model = EEGGCN(
        num_node_features=4,
        hidden_channels=64,
        num_classes=3,
        dropout=0.5,
    )

    out = model(graph.x, graph.edge_index, batch, graph.edge_attr)
    print("Output shape:", out.shape)
    print("Output:", out)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
