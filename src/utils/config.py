"""Configuration management using OmegaConf."""

from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file.
            **kwargs: Additional configuration parameters.
        """
        if config_path:
            self.cfg = OmegaConf.load(config_path)
        else:
            self.cfg = OmegaConf.create()
        
        # Override with kwargs
        if kwargs:
            override_cfg = OmegaConf.create(kwargs)
            self.cfg = OmegaConf.merge(self.cfg, override_cfg)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self.cfg, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self.cfg, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update.
        """
        override_cfg = OmegaConf.create(updates)
        self.cfg = OmegaConf.merge(self.cfg, override_cfg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return OmegaConf.to_container(self.cfg, resolve=True)
    
    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration file.
        """
        OmegaConf.save(self.cfg, path)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation.
        
        Args:
            key: Configuration key.
            
        Returns:
            Configuration value.
        """
        return self.cfg[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation.
        
        Args:
            key: Configuration key.
            value: Value to set.
        """
        self.cfg[key] = value
