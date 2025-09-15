import json
import os
from typing import Any, Dict


class ConfigManager:
    """
    统一管理GUI相关配置的加载、保存、校验和默认值。
    """

    def __init__(self, config_path: str, default_config: Dict[str, Any]):
        self.config_path = config_path
        self.default_config = default_config
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            self.save_config(self.default_config)
            return self.default_config.copy()
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 补全缺失项
        for k, v in self.default_config.items():
            if k not in data:
                data[k] = v
        return data

    def save_config(self, config: Dict[str, Any] = None):
        if config is None:
            config = self.config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value

    def update(self, new_config: Dict[str, Any]):
        self.config.update(new_config)

    def reset(self):
        self.config = self.default_config.copy()
        self.save_config()
