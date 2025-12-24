"""Dependency resolution and fetching."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml
from rich.console import Console

from .codebase_loader import CodebaseLoader, CodebaseSnapshot
from .repo_discovery import RepoDiscovery
from .version_fetcher import VersionFetcher

console = Console()


@dataclass
class Dependency:
    """Information about a dependency."""

    name: str
    version_spec: Optional[str]
    github_repo: Optional[str] = None
    is_python: bool = True


class DependencyResolver:
    """Handles dependency resolution and fetching."""

    # Mapping of common package names to their GitHub repos
    KNOWN_REPOS = {
        "torch": "pytorch/pytorch",
        "pytorch": "pytorch/pytorch",
        "tensorflow": "tensorflow/tensorflow",
        "jax": "google/jax",
        "flax": "google/flax",
        "numpy": "numpy/numpy",
        "scipy": "scipy/scipy",
        "pandas": "pandas-dev/pandas",
        "scikit-learn": "scikit-learn/scikit-learn",
        "sklearn": "scikit-learn/scikit-learn",
        "transformers": "huggingface/transformers",
        "datasets": "huggingface/datasets",
        "accelerate": "huggingface/accelerate",
        "peft": "huggingface/peft",
        "trl": "huggingface/trl",
        "diffusers": "huggingface/diffusers",
        "tokenizers": "huggingface/tokenizers",
        "safetensors": "huggingface/safetensors",
        "vllm": "vllm-project/vllm",
        "triton": "openai/triton",
        "flash-attn": "Dao-AILab/flash-attention",
        "flash_attn": "Dao-AILab/flash-attention",
        "xformers": "facebookresearch/xformers",
        "fairscale": "facebookresearch/fairscale",
        "deepspeed": "microsoft/DeepSpeed",
        "megatron": "NVIDIA/Megatron-LM",
        "apex": "NVIDIA/apex",
        "tensorrt": "NVIDIA/TensorRT",
        "onnx": "onnx/onnx",
        "onnxruntime": "microsoft/onnxruntime",
        "langchain": "langchain-ai/langchain",
        "llama-index": "run-llama/llama_index",
        "mlflow": "mlflow/mlflow",
        "ray": "ray-project/ray",
        "dask": "dask/dask",
        "wandb": "wandb/wandb",
        "optuna": "optuna/optuna",
        "hydra-core": "facebookresearch/hydra",
        "lightning": "Lightning-AI/pytorch-lightning",
        "pytorch-lightning": "Lightning-AI/pytorch-lightning",
        "einops": "arogozhnikov/einops",
        "sentencepiece": "google/sentencepiece",
        "tiktoken": "openai/tiktoken",
        "bitsandbytes": "TimDettmers/bitsandbytes",
    }

    def __init__(
        self,
        cache_dir: Path,
        github_token: Optional[str] = None,
        max_file_size: int = 1048576,
        max_codebase_size: int = 104857600,
    ):
        """
        Initialize the dependency resolver.

        Args:
            cache_dir: Directory to cache repositories
            github_token: GitHub API token
            max_file_size: Maximum file size to include
            max_codebase_size: Maximum total codebase size
        """
        self.cache_dir = cache_dir
        self.deps_dir = cache_dir / "deps"
        self.deps_dir.mkdir(parents=True, exist_ok=True)

        self.repo_discovery = RepoDiscovery(github_token)
        self.version_fetcher = VersionFetcher(cache_dir)
        self.codebase_loader = CodebaseLoader(max_file_size, max_codebase_size)

    def extract_dependencies(self, repo_path: Path) -> list[Dependency]:
        """
        Extract dependencies from a repository.

        Args:
            repo_path: Path to the repository

        Returns:
            List of dependencies
        """
        dependencies = []

        # Check for different dependency files
        if (repo_path / "pyproject.toml").exists():
            dependencies.extend(self._parse_pyproject(repo_path / "pyproject.toml"))

        if (repo_path / "setup.py").exists():
            dependencies.extend(self._parse_setup_py(repo_path / "setup.py"))

        if (repo_path / "requirements.txt").exists():
            dependencies.extend(self._parse_requirements(repo_path / "requirements.txt"))

        # Look for requirements in subdirectories
        for req_file in repo_path.glob("**/requirements*.txt"):
            if "test" not in str(req_file).lower() and "dev" not in str(req_file).lower():
                dependencies.extend(self._parse_requirements(req_file))

        # Deduplicate by name
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep.name.lower() not in seen:
                seen.add(dep.name.lower())
                unique_deps.append(dep)

        # Try to resolve GitHub repos for known packages
        for dep in unique_deps:
            if dep.github_repo is None:
                dep.github_repo = self.KNOWN_REPOS.get(dep.name.lower())

        return unique_deps

    def _parse_pyproject(self, path: Path) -> list[Dependency]:
        """Parse dependencies from pyproject.toml."""
        dependencies = []

        try:
            data = toml.load(path)

            # Check [project] dependencies
            if "project" in data and "dependencies" in data["project"]:
                for dep_str in data["project"]["dependencies"]:
                    dep = self._parse_dep_string(dep_str)
                    if dep:
                        dependencies.append(dep)

            # Check [tool.poetry.dependencies]
            if "tool" in data and "poetry" in data["tool"]:
                poetry_deps = data["tool"]["poetry"].get("dependencies", {})
                for name, spec in poetry_deps.items():
                    if name.lower() == "python":
                        continue
                    version = spec if isinstance(spec, str) else spec.get("version", "")
                    dependencies.append(Dependency(name=name, version_spec=version))

        except (toml.TomlDecodeError, OSError) as e:
            console.print(f"[yellow]Warning: Could not parse {path}: {e}[/yellow]")

        return dependencies

    def _parse_setup_py(self, path: Path) -> list[Dependency]:
        """Parse dependencies from setup.py (basic extraction)."""
        dependencies = []

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            # Look for install_requires
            match = re.search(
                r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL
            )
            if match:
                deps_str = match.group(1)
                for dep_str in re.findall(r"['\"]([^'\"]+)['\"]", deps_str):
                    dep = self._parse_dep_string(dep_str)
                    if dep:
                        dependencies.append(dep)

        except OSError as e:
            console.print(f"[yellow]Warning: Could not parse {path}: {e}[/yellow]")

        return dependencies

    def _parse_requirements(self, path: Path) -> list[Dependency]:
        """Parse dependencies from requirements.txt."""
        dependencies = []

        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue

                    # Handle -r references (skip for now)
                    if line.startswith("-r"):
                        continue

                    dep = self._parse_dep_string(line)
                    if dep:
                        dependencies.append(dep)

        except OSError as e:
            console.print(f"[yellow]Warning: Could not parse {path}: {e}[/yellow]")

        return dependencies

    def _parse_dep_string(self, dep_str: str) -> Optional[Dependency]:
        """Parse a dependency string like 'package>=1.0.0'."""
        # Remove comments
        dep_str = dep_str.split("#")[0].strip()

        # Skip URLs and local paths
        if dep_str.startswith(("http://", "https://", "git+", ".", "/")):
            return None

        # Handle extras like package[extra]
        dep_str = re.sub(r"\[.*?\]", "", dep_str)

        # Parse package name and version
        match = re.match(r"([a-zA-Z0-9_-]+)\s*(.*)", dep_str)
        if match:
            name = match.group(1)
            version_spec = match.group(2).strip() if match.group(2) else None
            return Dependency(name=name, version_spec=version_spec)

        return None

    def fetch_dependency_codebase(
        self,
        dependency: Dependency,
        file_patterns: Optional[list[str]] = None,
    ) -> Optional[CodebaseSnapshot]:
        """
        Fetch the codebase for a dependency.

        Uses the latest version of the dependency.

        Args:
            dependency: Dependency information
            file_patterns: Glob patterns for files to include

        Returns:
            CodebaseSnapshot of the dependency, or None if not available
        """
        if not dependency.github_repo:
            console.print(
                f"[yellow]No GitHub repo found for {dependency.name}, skipping[/yellow]"
            )
            return None

        try:
            # Get repo metadata
            metadata = self.repo_discovery.get_repo_metadata(dependency.github_repo)

            if not metadata.releases:
                console.print(
                    f"[yellow]No releases found for {dependency.name}, using default branch[/yellow]"
                )
                # Clone and use default branch
                repo_path = self.version_fetcher.clone_or_update_repo(
                    dependency.github_repo,
                    metadata.clone_url,
                )
                version = "main"
            else:
                # Use the latest release
                latest_release = metadata.releases[-1]
                repo_path = self.version_fetcher.clone_or_update_repo(
                    dependency.github_repo,
                    metadata.clone_url,
                )
                self.version_fetcher.checkout_version(repo_path, latest_release)
                version = latest_release.tag_name

            # Load the codebase (include everything, binary files become stubs)
            return self.codebase_loader.load_codebase(
                repo_path=repo_path,
                repo_name=dependency.github_repo,
                version=version,
                file_patterns=file_patterns or ["*"],
                exclude_patterns=[],  # Include all files
            )

        except Exception as e:
            console.print(
                f"[red]Error fetching dependency {dependency.name}: {e}[/red]"
            )
            return None

    def resolve_and_fetch_dependencies(
        self,
        repo_path: Path,
        depth: int = 1,
        max_deps: int = 10,
        file_patterns: Optional[list[str]] = None,
    ) -> list[CodebaseSnapshot]:
        """
        Resolve and fetch dependencies for a repository.

        Args:
            repo_path: Path to the repository
            depth: Dependency resolution depth (currently only 1 is supported)
            max_deps: Maximum number of dependencies to fetch
            file_patterns: Glob patterns for files to include

        Returns:
            List of dependency codebases
        """
        console.print(f"[cyan]Resolving dependencies for {repo_path.name}...[/cyan]")

        dependencies = self.extract_dependencies(repo_path)

        # Filter to only known dependencies with GitHub repos
        fetchable_deps = [d for d in dependencies if d.github_repo]

        console.print(
            f"[cyan]Found {len(dependencies)} dependencies, "
            f"{len(fetchable_deps)} can be fetched[/cyan]"
        )

        if len(fetchable_deps) > max_deps:
            console.print(
                f"[yellow]Limiting to {max_deps} most important dependencies[/yellow]"
            )
            # Prioritize well-known dependencies
            priority_order = list(self.KNOWN_REPOS.keys())
            fetchable_deps.sort(
                key=lambda d: (
                    priority_order.index(d.name.lower())
                    if d.name.lower() in priority_order
                    else 999
                )
            )
            fetchable_deps = fetchable_deps[:max_deps]

        # Fetch each dependency
        codebases = []
        for dep in fetchable_deps:
            console.print(f"[cyan]Fetching dependency: {dep.name}[/cyan]")
            codebase = self.fetch_dependency_codebase(dep, file_patterns)
            if codebase:
                codebases.append(codebase)

        console.print(f"[green]Fetched {len(codebases)} dependency codebases[/green]")
        return codebases

    def get_dependency_summary(self, repo_path: Path) -> str:
        """Get a summary of dependencies for a repository."""
        dependencies = self.extract_dependencies(repo_path)

        lines = ["# Dependencies", ""]
        for dep in dependencies:
            status = "✓" if dep.github_repo else "?"
            lines.append(f"{status} {dep.name}: {dep.version_spec or 'any'}")
            if dep.github_repo:
                lines.append(f"   GitHub: {dep.github_repo}")

        return "\n".join(lines)

