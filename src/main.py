"""Main orchestration and CLI for SFT dataset generation."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .auto_discover import AutoDiscover, CURATED_REPOS, ML_TOPICS, discover_and_save
from .codebase_loader import CodebaseLoader
from .config import Config, RepoConfig
from .dependency_resolver import DependencyResolver
from .diff_generator import DiffGenerator
from .repo_discovery import RepoDiscovery
from .sft_formatter import SFTExample, SFTFormatter
from .version_fetcher import VersionFetcher

console = Console()


class SFTDatasetGenerator:
    """Main orchestrator for SFT dataset generation."""

    def __init__(self, config: Config):
        """
        Initialize the dataset generator.

        Args:
            config: Configuration object
        """
        self.config = config
        config.ensure_dirs()

        # Initialize components
        self.repo_discovery = RepoDiscovery(config.github_token)
        self.version_fetcher = VersionFetcher(config.cache_dir)
        self.codebase_loader = CodebaseLoader(
            max_file_size=config.settings.max_file_size,
            max_codebase_size=config.settings.max_codebase_size,
        )
        self.dependency_resolver = DependencyResolver(
            cache_dir=config.cache_dir,
            github_token=config.github_token,
            max_file_size=config.settings.max_file_size,
            max_codebase_size=config.settings.max_codebase_size,
        )
        self.diff_generator = DiffGenerator()
        self.sft_formatter = SFTFormatter(config.output_dir)

    def process_repository(
        self,
        repo_config: RepoConfig,
        include_dependencies: bool = True,
    ) -> list[SFTExample]:
        """
        Process a single repository and generate SFT examples.

        Args:
            repo_config: Repository configuration
            include_dependencies: Whether to include dependency codebases

        Returns:
            List of SFT examples for this repository
        """
        console.print(
            Panel(
                f"[bold cyan]Processing {repo_config.name}[/bold cyan]\n"
                f"{repo_config.description}",
                title="Repository",
            )
        )

        examples = []

        try:
            # Get repository metadata and releases
            metadata = self.repo_discovery.get_repo_metadata(repo_config.name)

            if not metadata.releases:
                console.print(f"[yellow]No releases found for {repo_config.name}[/yellow]")
                return examples

            # Get version pairs
            version_pairs = self.repo_discovery.get_version_pairs(
                releases=metadata.releases,
                min_version=repo_config.min_version,
                max_pairs=repo_config.max_versions,
                skip_prereleases=True,
            )

            if not version_pairs:
                console.print(f"[yellow]No version pairs found for {repo_config.name}[/yellow]")
                return examples

            console.print(f"[cyan]Found {len(version_pairs)} version pairs to process[/cyan]")

            # Clone/update the repository
            repo_path = self.version_fetcher.clone_or_update_repo(
                repo_config.name,
                metadata.clone_url,
            )

            # Process each version pair
            for i, (from_release, to_release) in enumerate(version_pairs):
                console.print(
                    f"\n[cyan]Processing pair {i + 1}/{len(version_pairs)}: "
                    f"{from_release.tag_name} -> {to_release.tag_name}[/cyan]"
                )

                try:
                    example = self._process_version_pair(
                        repo_path=repo_path,
                        repo_config=repo_config,
                        from_release=from_release,
                        to_release=to_release,
                        include_dependencies=include_dependencies,
                    )
                    if example:
                        examples.append(example)
                        console.print(
                            f"[green]✓ Created example for "
                            f"{from_release.tag_name} -> {to_release.tag_name}[/green]"
                        )

                except Exception as e:
                    console.print(
                        f"[red]✗ Error processing {from_release.tag_name} -> "
                        f"{to_release.tag_name}: {e}[/red]"
                    )
                    continue

        except Exception as e:
            console.print(f"[red]Error processing repository {repo_config.name}: {e}[/red]")

        return examples

    def _process_version_pair(
        self,
        repo_path: Path,
        repo_config: RepoConfig,
        from_release,
        to_release,
        include_dependencies: bool,
    ) -> Optional[SFTExample]:
        """Process a single version pair."""
        # Checkout the "from" version
        self.version_fetcher.checkout_version(repo_path, from_release)

        # Load the codebase
        codebase = self.codebase_loader.load_codebase(
            repo_path=repo_path,
            repo_name=repo_config.name,
            version=from_release.tag_name,
            file_patterns=repo_config.file_patterns,
            exclude_patterns=repo_config.exclude_patterns,
        )

        # Load dependencies if requested
        num_dependencies = 0
        if include_dependencies and self.config.settings.include_dependencies:
            dep_codebases = self.dependency_resolver.resolve_and_fetch_dependencies(
                repo_path=repo_path,
                depth=self.config.settings.dependency_depth,
                max_deps=10,
                file_patterns=repo_config.file_patterns,
            )
            if dep_codebases:
                codebase = self.codebase_loader.merge_codebases(codebase, dep_codebases)
                num_dependencies = len(dep_codebases)

        # Generate the diff
        diff = self.diff_generator.generate_diff(
            repo_path=repo_path,
            from_release=from_release,
            to_release=to_release,
            file_patterns=repo_config.file_patterns,
            exclude_patterns=repo_config.exclude_patterns,
        )

        # Create the SFT example
        example = self.sft_formatter.create_example(
            repo_name=repo_config.name,
            from_version=from_release.tag_name,
            to_version=to_release.tag_name,
            codebase_text=codebase.to_text(),
            diff_text=diff.to_text(),
            num_files_in_codebase=len(codebase.files),
            num_files_changed=len(diff.file_diffs),
            num_dependencies=num_dependencies,
        )

        return example

    def generate_dataset(
        self,
        include_dependencies: bool = True,
    ) -> list[SFTExample]:
        """
        Generate the complete SFT dataset.

        Args:
            include_dependencies: Whether to include dependency codebases

        Returns:
            List of all SFT examples
        """
        all_examples = []

        console.print(
            Panel(
                f"[bold]SFT Dataset Generator[/bold]\n"
                f"Repositories: {len(self.config.repositories)}\n"
                f"Include Dependencies: {include_dependencies}",
                title="Starting Generation",
            )
        )

        for repo_config in self.config.repositories:
            examples = self.process_repository(
                repo_config=repo_config,
                include_dependencies=include_dependencies,
            )
            all_examples.extend(examples)

            console.print(
                f"[green]Completed {repo_config.name}: {len(examples)} examples[/green]\n"
            )

        return all_examples

    def save_dataset(
        self,
        examples: list[SFTExample],
        output_format: Optional[str] = None,
    ) -> list[Path]:
        """
        Save the dataset to disk.

        Args:
            examples: List of SFT examples
            output_format: Output format (jsonl, parquet, or both)

        Returns:
            List of saved file paths
        """
        output_format = output_format or self.config.settings.output_format
        saved_files = []

        if output_format in ("jsonl", "both"):
            # Save raw JSONL
            path = self.sft_formatter.save_examples_jsonl(examples, "dataset.jsonl")
            saved_files.append(path)

            # Save chat format
            path = self.sft_formatter.save_examples_chat_jsonl(examples, "dataset_chat.jsonl")
            saved_files.append(path)

        if output_format in ("parquet", "both"):
            path = self.sft_formatter.save_examples_parquet(examples, "dataset.parquet")
            saved_files.append(path)

        # Save statistics report
        stats_report = self.sft_formatter.generate_stats_report(examples)
        stats_path = self.config.output_dir / "stats.md"
        with open(stats_path, "w") as f:
            f.write(stats_report)
        saved_files.append(stats_path)

        # Save dataset card (README.md for HuggingFace)
        dataset_name = self.config.settings.get_hub_dataset_name()
        dataset_card = self.sft_formatter.generate_dataset_card(examples, dataset_name)
        card_path = self.config.output_dir / "README.md"
        with open(card_path, "w") as f:
            f.write(dataset_card)
        saved_files.append(card_path)

        return saved_files

    def print_summary(self, examples: list[SFTExample]) -> None:
        """Print a summary of the generated dataset."""
        from rich.panel import Panel
        from rich.table import Table

        if not examples:
            console.print("[yellow]No examples generated[/yellow]")
            return

        stats = self.sft_formatter.compute_statistics(examples)

        def fmt_size(size: float) -> str:
            if size >= 1_000_000_000:
                return f"{size / 1_000_000_000:.2f} GB"
            elif size >= 1_000_000:
                return f"{size / 1_000_000:.2f} MB"
            elif size >= 1_000:
                return f"{size / 1_000:.2f} KB"
            return f"{size:.0f} B"

        # Summary table
        table = Table(title="Dataset Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Input (Codebase)", justify="right")
        table.add_column("Output (Diff)", justify="right")

        table.add_row(
            "Mean Size",
            fmt_size(stats['input_size_mean']),
            fmt_size(stats['output_size_mean']),
        )
        table.add_row(
            "Min Size",
            fmt_size(stats['input_size_min']),
            fmt_size(stats['output_size_min']),
        )
        table.add_row(
            "Max Size",
            fmt_size(stats['input_size_max']),
            fmt_size(stats['output_size_max']),
        )
        table.add_row(
            "Mean Tokens",
            f"{stats['input_tokens_mean']:,.0f}",
            f"{stats['output_tokens_mean']:,.0f}",
        )

        console.print()
        console.print(Panel(
            f"[bold green]✓ Generated {stats['num_examples']} examples[/bold green]\n"
            f"From {stats['num_repos']} repositories\n"
            f"Total size: {fmt_size(stats['input_size_total'] + stats['output_size_total'])}",
            title="Dataset Complete",
        ))
        console.print(table)

        # Repo breakdown
        console.print("\n[bold]Per-Repository Breakdown:[/bold]")
        for repo, count in sorted(stats['repo_counts'].items()):
            console.print(f"  • {repo}: {count} examples")

    def upload_to_hub(
        self,
        examples: list[SFTExample],
        dataset_name: Optional[str] = None,
        repo_name: Optional[str] = None,
        private: bool = False,
    ) -> str:
        """Upload dataset to HuggingFace Hub."""
        if dataset_name is None:
            dataset_name = self.config.settings.get_hub_dataset_name(repo_name)
        return self.sft_formatter.upload_to_hub(examples, dataset_name, private)


# CLI Interface
@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="configs/repos.yaml",
    help="Path to configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
@click.option(
    "--cache",
    type=click.Path(),
    default=None,
    help="Cache directory for cloned repos",
)
@click.pass_context
def cli(ctx, config, output, cache):
    """SFT Codebase Dataset Generator

    Generate SFT training data from open-source repository version diffs.
    """
    ctx.ensure_object(dict)

    # Load configuration
    cfg = Config.from_yaml(config)
    cfg.output_dir = Path(output)
    if cache:
        cfg.cache_dir = Path(cache)

    ctx.obj["config"] = cfg
    ctx.obj["generator"] = SFTDatasetGenerator(cfg)


@cli.command()
@click.option(
    "--include-deps/--no-deps",
    default=True,
    help="Include dependency codebases",
)
@click.option(
    "--format",
    type=click.Choice(["jsonl", "parquet", "both"]),
    default="jsonl",
    help="Output format",
)
@click.option(
    "--upload/--no-upload",
    default=None,
    help="Upload to HuggingFace Hub (default: use config setting)",
)
@click.pass_context
def generate(ctx, include_deps, format, upload):
    """Generate the SFT dataset from configured repositories."""
    generator = ctx.obj["generator"]
    config = ctx.obj["config"]

    # Generate dataset
    examples = generator.generate_dataset(include_dependencies=include_deps)

    if not examples:
        console.print("[yellow]No examples generated[/yellow]")
        return

    # Save dataset
    saved_files = generator.save_dataset(examples, output_format=format)

    # Print summary statistics
    generator.print_summary(examples)

    console.print("\n[bold]Saved files:[/bold]")
    for path in saved_files:
        console.print(f"  • {path}")

    # Upload if requested (CLI flag overrides config)
    should_upload = upload if upload is not None else config.settings.upload_to_hub
    if should_upload:
        dataset_name = config.settings.get_hub_dataset_name()
        console.print(f"\n[cyan]Uploading to HuggingFace Hub as {dataset_name}...[/cyan]")
        url = generator.upload_to_hub(examples)
        console.print(f"[green]✓ Uploaded to: {url}[/green]")


@cli.command()
@click.argument("repo_name")
@click.option("--max-versions", "-n", type=int, default=5, help="Max version pairs")
@click.pass_context
def single(ctx, repo_name, max_versions):
    """Process a single repository."""
    generator = ctx.obj["generator"]

    # Create a temporary config for this repo
    repo_config = RepoConfig(
        name=repo_name,
        max_versions=max_versions,
    )

    examples = generator.process_repository(
        repo_config=repo_config,
        include_dependencies=True,
    )

    if examples:
        generator.save_dataset(examples)
        console.print(f"\n[green]Generated {len(examples)} examples for {repo_name}[/green]")
    else:
        console.print(f"[yellow]No examples generated for {repo_name}[/yellow]")


@cli.command()
@click.pass_context
def list_repos(ctx):
    """List configured repositories."""
    config = ctx.obj["config"]

    table = Table(title="Configured Repositories")
    table.add_column("Repository", style="cyan")
    table.add_column("Description")
    table.add_column("Patterns")

    for repo in config.repositories:
        table.add_row(
            repo.name,
            repo.description[:50] + "..." if len(repo.description) > 50 else repo.description,
            ", ".join(repo.file_patterns[:3]),
        )

    console.print(table)


@cli.command()
@click.argument("repo_name")
@click.pass_context
def releases(ctx, repo_name):
    """List releases for a repository."""
    generator = ctx.obj["generator"]

    metadata = generator.repo_discovery.get_repo_metadata(repo_name)

    table = Table(title=f"Releases for {repo_name}")
    table.add_column("Tag", style="cyan")
    table.add_column("Name")
    table.add_column("Date")
    table.add_column("Pre-release")

    for release in metadata.releases[-20:]:  # Last 20 releases
        table.add_row(
            release.tag_name,
            release.name[:40] + "..." if len(release.name) > 40 else release.name,
            release.published_at.strftime("%Y-%m-%d"),
            "Yes" if release.is_prerelease else "No",
        )

    console.print(table)
    console.print(f"\nTotal releases: {len(metadata.releases)}")


@cli.command()
@click.pass_context
def rate_limit(ctx):
    """Check GitHub API rate limit status."""
    generator = ctx.obj["generator"]

    limits = generator.repo_discovery.check_rate_limit()

    table = Table(title="GitHub API Rate Limits")
    table.add_column("Resource")
    table.add_column("Remaining")
    table.add_column("Limit")

    table.add_row(
        "Core API",
        str(limits["core_remaining"]),
        str(limits["core_limit"]),
    )
    table.add_row(
        "Search API",
        str(limits["search_remaining"]),
        str(limits["search_limit"]),
    )

    console.print(table)

    if limits["core_remaining"] < 100:
        console.print(
            "\n[yellow]Warning: Low rate limit remaining. "
            "Consider setting GITHUB_TOKEN environment variable.[/yellow]"
        )


@cli.command()
@click.argument("query", default="machine learning")
@click.option("--language", "-l", default="Python", help="Programming language")
@click.option("--min-stars", "-s", type=int, default=1000, help="Minimum stars")
@click.option("--max-results", "-n", type=int, default=10, help="Max results")
@click.pass_context
def search(ctx, query, language, min_stars, max_results):
    """Search for popular repositories."""
    generator = ctx.obj["generator"]

    repos = generator.repo_discovery.search_popular_repos(
        query=query,
        language=language,
        min_stars=min_stars,
        max_results=max_results,
    )

    console.print(f"\n[bold]Search results for '{query}':[/bold]\n")
    for repo in repos:
        console.print(f"  - {repo}")


@cli.command()
@click.option("--output", "-o", default="configs/repos.yaml", help="Output YAML path")
@click.option("--curated/--no-curated", default=True, help="Include curated ML repos")
@click.option("--topics/--no-topics", default=True, help="Discover by GitHub topics")
@click.option("--min-stars", "-s", type=int, default=1000, help="Minimum stars")
@click.option("--min-releases", "-r", type=int, default=2, help="Minimum releases")
@click.pass_context
def discover(ctx, output, curated, topics, min_stars, min_releases):
    """Auto-discover ML/AI repositories and generate config.
    
    This searches GitHub for popular ML/AI repositories with releases
    and generates a repos.yaml configuration file.
    """
    config = ctx.obj["config"]
    
    repos = discover_and_save(
        output_path=output,
        use_curated=curated,
        discover_topics=topics,
        min_stars=min_stars,
        min_releases=min_releases,
        github_token=config.github_token,
    )
    
    console.print(f"\n[green]Discovered {len(repos)} repositories[/green]")
    console.print(f"[green]Configuration saved to {output}[/green]")


@cli.command()
@click.option("--topic", "-t", multiple=True, help="Specific topics to search")
@click.option("--min-stars", "-s", type=int, default=500, help="Minimum stars")
@click.option("--max-results", "-n", type=int, default=20, help="Max results per topic")
@click.pass_context
def browse_topics(ctx, topic, min_stars, max_results):
    """Browse repositories by GitHub topic.
    
    Examples:
        sft-dataset browse-topics -t llm -t transformers
        sft-dataset browse-topics --min-stars 5000
    """
    config = ctx.obj["config"]
    discoverer = AutoDiscover(config.github_token)
    
    topics_to_search = list(topic) if topic else ML_TOPICS[:5]
    
    for t in topics_to_search:
        repos = discoverer.discover_by_topic(
            topic=t,
            min_stars=min_stars,
            max_results=max_results,
        )
        discoverer.display_repos(repos, title=f"Topic: {t}")
        console.print()


@cli.command()
@click.pass_context
def list_curated(ctx):
    """List all curated ML/AI repositories.
    
    These are hand-picked high-quality repositories that are known
    to have good release practices and valuable training data.
    """
    console.print("\n[bold]Curated ML/AI Repositories:[/bold]\n")
    
    categories = {
        "Core ML Frameworks": ["pytorch/pytorch", "tensorflow/tensorflow", "google/jax", "apple/mlx"],
        "HuggingFace Ecosystem": [r for r in CURATED_REPOS if "huggingface" in r],
        "Inference Engines": ["vllm-project/vllm", "NVIDIA/TensorRT-LLM", "ggerganov/llama.cpp", "mlc-ai/mlc-llm"],
        "LLM Frameworks": ["langchain-ai/langchain", "run-llama/llama_index", "BerriAI/litellm"],
        "Training & Fine-tuning": ["hiyouga/LLaMA-Factory", "OpenAccess-AI-Collective/axolotl", "microsoft/DeepSpeed"],
    }
    
    for category, repos in categories.items():
        console.print(f"[cyan]{category}:[/cyan]")
        for repo in repos:
            if repo in CURATED_REPOS:
                console.print(f"  • {repo}")
        console.print()
    
    console.print(f"[dim]Total curated repos: {len(CURATED_REPOS)}[/dim]")
    console.print("\n[dim]Run 'sft-dataset discover --curated' to fetch metadata and generate config[/dim]")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()

