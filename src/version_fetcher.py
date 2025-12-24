"""Version fetching and git operations."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from git import Repo
from git.exc import GitCommandError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .repo_discovery import ReleaseInfo

console = Console()


class VersionFetcher:
    """Handles fetching specific versions of repositories."""

    def __init__(self, cache_dir: Path):
        """
        Initialize the version fetcher.

        Args:
            cache_dir: Directory to cache cloned repositories
        """
        self.cache_dir = cache_dir
        self.repos_dir = cache_dir / "repos"
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    def clone_or_update_repo(
        self,
        repo_name: str,
        clone_url: str,
    ) -> Path:
        """
        Clone a repository or update if it already exists.

        Args:
            repo_name: Repository name in format "owner/repo"
            clone_url: Git clone URL

        Returns:
            Path to the cloned repository
        """
        repo_path = self.repos_dir / repo_name.replace("/", "_")

        if repo_path.exists():
            console.print(f"[cyan]Updating existing clone of {repo_name}...[/cyan]")
            try:
                repo = Repo(repo_path)
                origin = repo.remotes.origin
                origin.fetch(tags=True, force=True)
                console.print(f"[green]Updated {repo_name}[/green]")
            except GitCommandError as e:
                console.print(f"[yellow]Failed to update, re-cloning: {e}[/yellow]")
                shutil.rmtree(repo_path)
                return self._clone_repo(repo_name, clone_url, repo_path)
        else:
            return self._clone_repo(repo_name, clone_url, repo_path)

        return repo_path

    def _clone_repo(self, repo_name: str, clone_url: str, repo_path: Path) -> Path:
        """Clone a repository."""
        console.print(f"[cyan]Cloning {repo_name}...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Cloning {repo_name}...", total=None)

            try:
                Repo.clone_from(
                    clone_url,
                    repo_path,
                    multi_options=["--filter=blob:none"],  # Partial clone for speed
                )
                console.print(f"[green]Cloned {repo_name}[/green]")
            except GitCommandError as e:
                raise RuntimeError(f"Failed to clone {repo_name}: {e}") from e

        return repo_path

    def checkout_version(
        self,
        repo_path: Path,
        release: ReleaseInfo,
    ) -> Path:
        """
        Checkout a specific version of a repository.

        Args:
            repo_path: Path to the cloned repository
            release: Release information

        Returns:
            Path to the checked out repository
        """
        repo = Repo(repo_path)

        console.print(f"[cyan]Checking out {release.tag_name}...[/cyan]")

        try:
            # First try to checkout by tag
            repo.git.checkout(release.tag_name, force=True)
        except GitCommandError:
            # Fall back to commit SHA
            try:
                repo.git.checkout(release.commit_sha, force=True)
            except GitCommandError as e:
                raise RuntimeError(
                    f"Failed to checkout {release.tag_name} ({release.commit_sha}): {e}"
                ) from e

        # Clean any untracked files
        repo.git.clean("-fdx")

        console.print(f"[green]Checked out {release.tag_name}[/green]")
        return repo_path

    def get_version_snapshot(
        self,
        repo_path: Path,
        release: ReleaseInfo,
        output_dir: Path,
    ) -> Path:
        """
        Create a snapshot of a specific version in a separate directory.

        Args:
            repo_path: Path to the cloned repository
            release: Release information
            output_dir: Directory to store the snapshot

        Returns:
            Path to the snapshot directory
        """
        snapshot_dir = output_dir / f"{repo_path.name}_{release.tag_name}"

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

        # Checkout the version
        self.checkout_version(repo_path, release)

        # Copy the repository (excluding .git)
        shutil.copytree(
            repo_path,
            snapshot_dir,
            ignore=shutil.ignore_patterns(".git"),
        )

        return snapshot_dir

    def get_file_at_version(
        self,
        repo_path: Path,
        file_path: str,
        commit_sha: str,
    ) -> Optional[str]:
        """
        Get the contents of a file at a specific commit.

        Args:
            repo_path: Path to the cloned repository
            file_path: Relative path to the file
            commit_sha: Git commit SHA

        Returns:
            File contents as string, or None if file doesn't exist
        """
        repo = Repo(repo_path)

        try:
            commit = repo.commit(commit_sha)
            blob = commit.tree / file_path
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except (KeyError, GitCommandError):
            return None

    def get_tags(self, repo_path: Path) -> list[str]:
        """Get all tags in a repository."""
        repo = Repo(repo_path)
        return [tag.name for tag in repo.tags]

    def get_commit_for_tag(self, repo_path: Path, tag_name: str) -> Optional[str]:
        """Get the commit SHA for a tag."""
        repo = Repo(repo_path)

        try:
            tag = repo.tags[tag_name]
            # Handle annotated tags
            if hasattr(tag.tag, "object"):
                return tag.tag.object.hexsha
            return tag.commit.hexsha
        except (IndexError, KeyError):
            return None

    def shallow_clone_at_tag(
        self,
        repo_name: str,
        clone_url: str,
        tag: str,
        output_dir: Path,
    ) -> Path:
        """
        Create a shallow clone at a specific tag.

        This is more efficient for one-off version fetches.

        Args:
            repo_name: Repository name
            clone_url: Git clone URL
            tag: Tag name to clone
            output_dir: Directory to clone into

        Returns:
            Path to the cloned repository
        """
        repo_path = output_dir / f"{repo_name.replace('/', '_')}_{tag}"

        if repo_path.exists():
            shutil.rmtree(repo_path)

        console.print(f"[cyan]Shallow cloning {repo_name} at {tag}...[/cyan]")

        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    tag,
                    clone_url,
                    str(repo_path),
                ],
                check=True,
                capture_output=True,
            )
            console.print(f"[green]Cloned {repo_name} at {tag}[/green]")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone {repo_name} at {tag}: {e.stderr.decode()}") from e

        return repo_path

