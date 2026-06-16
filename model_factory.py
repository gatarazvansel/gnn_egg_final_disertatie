from src.models.gcn_model import EEGGCN
from src.models.gat_model import EEGGAT


SUPPORTED_MODELS = ("GCN", "GAT")


def normalize_model_name(model_name: str) -> str:
    if model_name is None:
        raise ValueError("model_name cannot be None.")

    normalized = str(model_name).strip().upper()

    if normalized not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {SUPPORTED_MODELS}"
        )

    return normalized


def get_supported_models() -> list[str]:
    return list(SUPPORTED_MODELS)


def get_model(
    model_name: str,
    num_node_features: int = 4,
    hidden_channels: int = 64,
    num_classes: int = 2,
    dropout: float = 0.5,
    gat_heads_first_layer: int = 4,
    gat_heads_second_layer: int = 1,
):
    model_name = normalize_model_name(model_name)

    if model_name == "GCN":
        return EEGGCN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    if model_name == "GAT":
        return EEGGAT(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            heads_first_layer=gat_heads_first_layer,
            heads_second_layer=gat_heads_second_layer,
            dropout=dropout,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: {SUPPORTED_MODELS}"
    )