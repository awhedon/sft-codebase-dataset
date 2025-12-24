"""Automated repository discovery for ML/AI projects."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import yaml
from github import Github
from github.GithubException import GithubException, RateLimitExceededException
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class DiscoveredRepo:
    """A discovered repository with metadata."""

    name: str
    description: str
    stars: int
    language: str
    topics: list[str]
    last_release: Optional[str]
    release_count: int
    last_updated: datetime
    url: str


# Curated list of high-quality ML/AI topics to search
ML_TOPICS = [
    "large-language-models",
    "llm",
    "transformers",
    "deep-learning",
    "machine-learning",
    "pytorch",
    "tensorflow",
    "nlp",
    "natural-language-processing",
    "computer-vision",
    "generative-ai",
    "diffusion-models",
    "reinforcement-learning",
    "mlops",
    "model-serving",
    "inference",
    "quantization",
    "fine-tuning",
    "rag",
    "vector-database",
    "embeddings",
]

# Known high-quality repos that should always be included
CURATED_REPOS = [
    # Core ML Frameworks
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "google/jax",
    "apple/mlx",
    
    # HuggingFace Ecosystem
    "huggingface/transformers",
    "huggingface/diffusers",
    "huggingface/accelerate",
    "huggingface/peft",
    "huggingface/trl",
    "huggingface/datasets",
    "huggingface/tokenizers",
    "huggingface/safetensors",
    
    # Inference Engines
    "vllm-project/vllm",
    "NVIDIA/TensorRT-LLM",
    "ggerganov/llama.cpp",
    "ggerganov/whisper.cpp",
    "mlc-ai/mlc-llm",
    "xorbitsai/inference",
    
    # Training & Fine-tuning
    "hiyouga/LLaMA-Factory",
    "OpenAccess-AI-Collective/axolotl",
    "Lightning-AI/pytorch-lightning",
    "microsoft/DeepSpeed",
    "facebookresearch/fairscale",
    
    # LLM Frameworks
    "langchain-ai/langchain",
    "run-llama/llama_index",
    "BerriAI/litellm",
    "guidance-ai/guidance",
    "outlines-dev/outlines",
    
    # Compilers & Optimization
    "openai/triton",
    "NVIDIA/cutlass",
    "Dao-AILab/flash-attention",
    "facebookresearch/xformers",
    
    # MLOps
    "mlflow/mlflow",
    "ray-project/ray",
    "wandb/wandb",
    "bentoml/BentoML",
    
    # Model Implementations
    "meta-llama/llama",
    "meta-llama/llama-models",
    "mistralai/mistral-src",
    "QwenLM/Qwen",
    "THUDM/ChatGLM-6B",
    "lm-sys/FastChat",
    
    # Agents & Assistants
    "microsoft/autogen",
    "Significant-Gravitas/AutoGPT",
    "geekan/MetaGPT",
    
    # Evaluation & Benchmarks
    "EleutherAI/lm-evaluation-harness",
    "openai/evals",
]


class AutoDiscover:
    """Automated repository discovery."""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token."""
        self.github = Github(github_token) if github_token else Github()

    def discover_by_topic(
        self,
        topic: str,
        min_stars: int = 500,
        min_releases: int = 1,
        max_results: int = 20,
        language: Optional[str] = None,
    ) -> list[DiscoveredRepo]:
        """
        Discover repositories by GitHub topic.

        Args:
            topic: GitHub topic to search
            min_stars: Minimum star count
            min_releases: Minimum number of releases
            max_results: Maximum results to return
            language: Filter by programming language

        Returns:
            List of discovered repositories
        """
        console.print(f"[cyan]Searching topic: {topic}...[/cyan]")

        query = f"topic:{topic} stars:>={min_stars}"
        if language:
            query += f" language:{language}"

        try:
            repos = self.github.search_repositories(
                query=query,
                sort="stars",
                order="desc",
            )
        except RateLimitExceededException:
            console.print("[red]Rate limit exceeded. Set GITHUB_TOKEN for higher limits.[/red]")
            return []

        discovered = []
        count = 0

        for repo in repos:
            if count >= max_results:
                break

            try:
                # Check release count
                releases = list(repo.get_releases()[:10])  # Check first 10
                if len(releases) < min_releases:
                    continue

                last_release = releases[0].tag_name if releases else None

                discovered.append(
                    DiscoveredRepo(
                        name=repo.full_name,
                        description=repo.description or "",
                        stars=repo.stargazers_count,
                        language=repo.language or "Unknown",
                        topics=repo.topics or [],
                        last_release=last_release,
                        release_count=len(releases),
                        last_updated=repo.updated_at,
                        url=repo.html_url,
                    )
                )
                count += 1

            except GithubException:
                continue

        return discovered

    def discover_all_topics(
        self,
        topics: Optional[list[str]] = None,
        min_stars: int = 1000,
        min_releases: int = 2,
        max_per_topic: int = 10,
    ) -> list[DiscoveredRepo]:
        """
        Discover repositories across multiple topics.

        Args:
            topics: List of topics to search (defaults to ML_TOPICS)
            min_stars: Minimum star count
            min_releases: Minimum number of releases
            max_per_topic: Max repos per topic

        Returns:
            Deduplicated list of discovered repositories
        """
        topics = topics or ML_TOPICS
        all_repos: dict[str, DiscoveredRepo] = {}

        for topic in topics:
            try:
                repos = self.discover_by_topic(
                    topic=topic,
                    min_stars=min_stars,
                    min_releases=min_releases,
                    max_results=max_per_topic,
                )
                for repo in repos:
                    if repo.name not in all_repos:
                        all_repos[repo.name] = repo

            except RateLimitExceededException:
                console.print("[red]Rate limit exceeded, stopping discovery.[/red]")
                break

        # Sort by stars
        return sorted(all_repos.values(), key=lambda r: r.stars, reverse=True)

    def get_curated_repos(
        self,
        repos: Optional[list[str]] = None,
    ) -> list[DiscoveredRepo]:
        """
        Get metadata for curated repository list.

        Args:
            repos: List of repo names (defaults to CURATED_REPOS)

        Returns:
            List of discovered repositories with metadata
        """
        repos = repos or CURATED_REPOS
        discovered = []

        console.print(f"[cyan]Fetching metadata for {len(repos)} curated repos...[/cyan]")

        for repo_name in repos:
            try:
                repo = self.github.get_repo(repo_name)
                releases = list(repo.get_releases()[:10])

                discovered.append(
                    DiscoveredRepo(
                        name=repo.full_name,
                        description=repo.description or "",
                        stars=repo.stargazers_count,
                        language=repo.language or "Unknown",
                        topics=repo.topics or [],
                        last_release=releases[0].tag_name if releases else None,
                        release_count=len(releases),
                        last_updated=repo.updated_at,
                        url=repo.html_url,
                    )
                )

            except GithubException as e:
                console.print(f"[yellow]Could not fetch {repo_name}: {e}[/yellow]")
                continue

        return discovered

    def filter_repos(
        self,
        repos: list[DiscoveredRepo],
        min_stars: int = 0,
        min_releases: int = 0,
        max_age_days: Optional[int] = None,
        languages: Optional[list[str]] = None,
        required_topics: Optional[list[str]] = None,
    ) -> list[DiscoveredRepo]:
        """
        Filter discovered repositories.

        Args:
            repos: List of repos to filter
            min_stars: Minimum star count
            min_releases: Minimum release count
            max_age_days: Max days since last update
            languages: Allowed languages
            required_topics: Required topics (any match)

        Returns:
            Filtered list of repositories
        """
        filtered = []

        for repo in repos:
            if repo.stars < min_stars:
                continue
            if repo.release_count < min_releases:
                continue
            if max_age_days:
                age = datetime.now() - repo.last_updated.replace(tzinfo=None)
                if age > timedelta(days=max_age_days):
                    continue
            if languages and repo.language not in languages:
                continue
            if required_topics:
                if not any(t in repo.topics for t in required_topics):
                    continue

            filtered.append(repo)

        return filtered

    def generate_yaml_config(
        self,
        repos: list[DiscoveredRepo],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate YAML configuration from discovered repos.

        Args:
            repos: List of discovered repositories
            output_path: Optional path to save YAML

        Returns:
            YAML string
        """
        config = {
            "repositories": [
                {
                    "name": repo.name,
                    "description": repo.description[:100] if repo.description else "",
                }
                for repo in repos
            ],
            "settings": {
                "max_file_size": 1048576,
                "max_codebase_size": 524288000,
                "include_dependencies": True,
                "dependency_depth": 1,
                "output_format": "jsonl",
                "upload_to_hub": False,
                "hub_dataset_name": "sft-codebase-diffs",
            },
        }

        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

        if output_path:
            with open(output_path, "w") as f:
                f.write("# Auto-generated repository configuration\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total repositories: {len(repos)}\n\n")
                f.write(yaml_str)
            console.print(f"[green]Saved configuration to {output_path}[/green]")

        return yaml_str

    def display_repos(self, repos: list[DiscoveredRepo], title: str = "Discovered Repositories"):
        """Display repositories in a table."""
        table = Table(title=title)
        table.add_column("Repository", style="cyan", max_width=40)
        table.add_column("Stars", justify="right")
        table.add_column("Releases", justify="right")
        table.add_column("Language")
        table.add_column("Last Release")

        for repo in repos[:50]:  # Show top 50
            table.add_row(
                repo.name,
                f"{repo.stars:,}",
                str(repo.release_count),
                repo.language,
                repo.last_release or "N/A",
            )

        console.print(table)
        if len(repos) > 50:
            console.print(f"[dim]... and {len(repos) - 50} more[/dim]")


def discover_and_save(
    output_path: str = "configs/repos.yaml",
    use_curated: bool = True,
    discover_topics: bool = True,
    min_stars: int = 1000,
    min_releases: int = 2,
    github_token: Optional[str] = None,
) -> list[DiscoveredRepo]:
    """
    Discover repositories and save configuration.

    Args:
        output_path: Path to save YAML config
        use_curated: Include curated repo list
        discover_topics: Discover by topics
        min_stars: Minimum stars for topic discovery
        min_releases: Minimum releases
        github_token: GitHub API token

    Returns:
        List of all discovered repositories
    """
    discoverer = AutoDiscover(github_token)
    all_repos: dict[str, DiscoveredRepo] = {}

    # Get curated repos
    if use_curated:
        console.print("\n[bold]Fetching curated repositories...[/bold]")
        curated = discoverer.get_curated_repos()
        for repo in curated:
            all_repos[repo.name] = repo
        console.print(f"[green]Found {len(curated)} curated repos[/green]")

    # Discover by topics
    if discover_topics:
        console.print("\n[bold]Discovering repositories by topic...[/bold]")
        discovered = discoverer.discover_all_topics(
            min_stars=min_stars,
            min_releases=min_releases,
        )
        for repo in discovered:
            if repo.name not in all_repos:
                all_repos[repo.name] = repo
        console.print(f"[green]Discovered {len(discovered)} additional repos[/green]")

    # Filter to repos with releases
    repos = [r for r in all_repos.values() if r.release_count >= min_releases]
    repos.sort(key=lambda r: r.stars, reverse=True)

    console.print(f"\n[bold]Total: {len(repos)} repositories with releases[/bold]")

    # Display and save
    discoverer.display_repos(repos)
    discoverer.generate_yaml_config(repos, output_path)

    return repos

