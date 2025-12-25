"""
GNN model configuration.

REMINDER: This project is for research and education only.
"""

from pydantic import BaseModel, Field


class GNNConfig(BaseModel):
    """Configuration for the GNN encoder."""
    
    # Architecture
    hidden_dim: int = Field(default=128, description="Hidden layer dimension")
    embedding_dim: int = Field(default=64, description="Final embedding dimension")
    num_layers: int = Field(default=3, description="Number of GNN layers")
    num_heads: int = Field(default=4, description="Number of attention heads")
    
    # Regularization
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    attention_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Activation
    activation: str = Field(default="elu", pattern="^(relu|elu|gelu|leaky_relu)$")
    
    # Normalization
    layer_norm: bool = Field(default=True)
    batch_norm: bool = Field(default=False)


class TrainingConfig(BaseModel):
    """Configuration for GNN training."""
    
    # Optimization
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    
    # Training parameters
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=32, gt=0)
    patience: int = Field(default=15, gt=0, description="Early stopping patience")
    
    # Device
    device: str = Field(default="auto")
    seed: int = Field(default=42)


class PretrainingConfig(BaseModel):
    """Configuration for self-supervised pretraining."""
    
    enabled: bool = Field(default=True)
    
    # Tasks
    edge_prediction: bool = Field(default=True)
    node_attribute_prediction: bool = Field(default=True)
    masked_node_prediction: bool = Field(default=True)
    
    # Loss weights
    edge_pred_weight: float = Field(default=1.0)
    node_attr_weight: float = Field(default=0.5)
    masked_node_weight: float = Field(default=0.5)
    
    # Masking
    mask_ratio: float = Field(default=0.15, ge=0, le=1)
    negative_sampling_ratio: int = Field(default=5, gt=0)

