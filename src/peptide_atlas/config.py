"""
Global configuration for the Peptide Atlas.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""
    
    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
    )
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("outputs"))
    config_dir: Path = Field(default=Path("configs"))
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Device settings
    device: str = Field(default="auto")  # auto, cpu, cuda, mps
    seed: int = Field(default=42)
    
    # Visualization
    plotly_renderer: str = Field(default="browser")
    
    class Config:
        env_prefix = "PEPTIDE_ATLAS_"
        env_file = ".env"
        extra = "ignore"
    
    @property
    def data_path(self) -> Path:
        """Get absolute data directory path."""
        if self.data_dir.is_absolute():
            return self.data_dir
        return self.project_root / self.data_dir
    
    @property
    def output_path(self) -> Path:
        """Get absolute output directory path."""
        if self.output_dir.is_absolute():
            return self.output_dir
        return self.project_root / self.output_dir
    
    @property
    def config_path(self) -> Path:
        """Get absolute config directory path."""
        if self.config_dir.is_absolute():
            return self.config_dir
        return self.project_root / self.config_dir
    
    def get_device(self) -> str:
        """Get the compute device to use."""
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def configure_logging() -> None:
    """Configure logging for the application."""
    from loguru import logger
    import sys
    
    # Remove default handler
    logger.remove()
    
    # Add configured handler
    logger.add(
        sys.stderr,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True,
    )
    
    # Log disclaimer on startup
    logger.info("Peptide Atlas initialized")
    logger.warning(
        "RESEARCH USE ONLY - This tool provides NO medical advice, "
        "NO dosing information, NO therapeutic recommendations."
    )

