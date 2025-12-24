"""Configuration management for the SFT dataset generator."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RepoConfig:
    """Configuration for a single repository."""

    name: str
    description: str = ""
    min_version: Optional[str] = None
    max_versions: Optional[int] = None
    file_patterns: list[str] = field(default_factory=lambda: ["*.py", "*.cpp", "*.h", "*.cu"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["tests/*", "docs/*"])

    @property
    def owner(self) -> str:
        return self.name.split("/")[0]

    @property
    def repo(self) -> str:
        return self.name.split("/")[1]


@dataclass
class Settings:
    """Global settings for the dataset generator."""

    max_file_size: int = 1048576  # 1MB
    max_codebase_size: int = 104857600  # 100MB
    include_dependencies: bool = True
    dependency_depth: int = 1
    output_format: str = "jsonl"
    upload_to_hub: bool = False
    hub_dataset_name: str = "sft-codebase-diffs"


@dataclass
class Config:
    """Main configuration container."""

    repositories: list[RepoConfig]
    settings: Settings
    github_token: Optional[str] = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "sft-codebase-dataset")
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        repositories = [
            RepoConfig(
                name=repo["name"],
                description=repo.get("description", ""),
                min_version=repo.get("min_version"),
                max_versions=repo.get("max_versions"),
                file_patterns=repo.get("file_patterns", ["*.py", "*.cpp", "*.h", "*.cu"]),
                exclude_patterns=repo.get("exclude_patterns", ["tests/*", "docs/*"]),
            )
            for repo in data.get("repositories", [])
        ]

        settings_data = data.get("settings", {})
        settings = Settings(
            max_file_size=settings_data.get("max_file_size", 1048576),
            max_codebase_size=settings_data.get("max_codebase_size", 104857600),
            include_dependencies=settings_data.get("include_dependencies", True),
            dependency_depth=settings_data.get("dependency_depth", 1),
            output_format=settings_data.get("output_format", "jsonl"),
            upload_to_hub=settings_data.get("upload_to_hub", False),
            hub_dataset_name=settings_data.get("hub_dataset_name", "sft-codebase-diffs"),
        )

        github_token = os.environ.get("GITHUB_TOKEN")

        return cls(
            repositories=repositories,
            settings=settings,
            github_token=github_token,
        )

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "repos").mkdir(exist_ok=True)
        (self.cache_dir / "deps").mkdir(exist_ok=True)

