"""Repository discovery and metadata fetching."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from github import Github
from github.GithubException import GithubException
from rich.console import Console

console = Console()


@dataclass
class ReleaseInfo:
    """Information about a repository release."""

    tag_name: str
    name: str
    published_at: datetime
    tarball_url: str
    commit_sha: str
    is_prerelease: bool

    def __lt__(self, other: "ReleaseInfo") -> bool:
        """Compare releases by publication date."""
        return self.published_at < other.published_at


@dataclass
class RepoMetadata:
    """Metadata about a repository."""

    owner: str
    name: str
    full_name: str
    description: str
    default_branch: str
    clone_url: str
    stars: int
    language: str
    releases: list[ReleaseInfo]


class RepoDiscovery:
    """Handles repository discovery and metadata fetching."""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for higher rate limits."""
        self.github = Github(github_token) if github_token else Github()

    def get_repo_metadata(self, repo_name: str) -> RepoMetadata:
        """
        Fetch metadata for a repository.

        Args:
            repo_name: Repository name in format "owner/repo"

        Returns:
            RepoMetadata object with repository information
        """
        console.print(f"[cyan]Fetching metadata for {repo_name}...[/cyan]")

        try:
            repo = self.github.get_repo(repo_name)
        except GithubException as e:
            raise ValueError(f"Failed to fetch repository {repo_name}: {e}") from e

        releases = self._get_releases(repo)

        return RepoMetadata(
            owner=repo.owner.login,
            name=repo.name,
            full_name=repo.full_name,
            description=repo.description or "",
            default_branch=repo.default_branch,
            clone_url=repo.clone_url,
            stars=repo.stargazers_count,
            language=repo.language or "Unknown",
            releases=releases,
        )

    def _get_releases(self, repo) -> list[ReleaseInfo]:
        """Fetch all releases for a repository."""
        releases = []

        console.print(f"[cyan]Fetching releases for {repo.full_name}...[/cyan]")

        for release in repo.get_releases():
            try:
                # Get the commit SHA for this tag
                tag_ref = repo.get_git_ref(f"tags/{release.tag_name}")
                commit_sha = tag_ref.object.sha

                # Handle annotated tags (tag object points to commit)
                if tag_ref.object.type == "tag":
                    tag_obj = repo.get_git_tag(tag_ref.object.sha)
                    commit_sha = tag_obj.object.sha

                releases.append(
                    ReleaseInfo(
                        tag_name=release.tag_name,
                        name=release.title or release.tag_name,
                        published_at=release.published_at,
                        tarball_url=release.tarball_url,
                        commit_sha=commit_sha,
                        is_prerelease=release.prerelease,
                    )
                )
            except GithubException:
                # Skip releases where we can't get the commit SHA
                console.print(
                    f"[yellow]Warning: Could not get commit SHA for {release.tag_name}[/yellow]"
                )
                continue

        # Sort by publication date
        releases.sort()

        console.print(f"[green]Found {len(releases)} releases[/green]")
        return releases

    def get_version_pairs(
        self,
        releases: list[ReleaseInfo],
        min_version: Optional[str] = None,
        max_pairs: Optional[int] = None,
        skip_prereleases: bool = True,
    ) -> list[tuple[ReleaseInfo, ReleaseInfo]]:
        """
        Get consecutive release pairs for diff generation.

        Args:
            releases: List of releases sorted by date
            min_version: Minimum version to start from (optional)
            max_pairs: Maximum number of pairs to return (optional)
            skip_prereleases: Whether to skip pre-release versions

        Returns:
            List of (current_version, next_version) tuples
        """
        if skip_prereleases:
            releases = [r for r in releases if not r.is_prerelease]

        if min_version:
            # Find the index of min_version
            start_idx = 0
            for i, r in enumerate(releases):
                if self._version_matches(r.tag_name, min_version):
                    start_idx = i
                    break
            releases = releases[start_idx:]

        pairs = []
        for i in range(len(releases) - 1):
            pairs.append((releases[i], releases[i + 1]))

        if max_pairs:
            pairs = pairs[:max_pairs]

        return pairs

    def _version_matches(self, tag: str, version: str) -> bool:
        """Check if a tag matches a version string."""
        # Normalize both strings
        tag_normalized = re.sub(r"^v", "", tag.lower())
        version_normalized = re.sub(r"^v", "", version.lower())
        return tag_normalized.startswith(version_normalized)

    def search_popular_repos(
        self,
        query: str = "machine learning",
        language: str = "Python",
        min_stars: int = 1000,
        max_results: int = 10,
    ) -> list[str]:
        """
        Search for popular repositories matching criteria.

        Args:
            query: Search query
            language: Programming language filter
            min_stars: Minimum star count
            max_results: Maximum number of results

        Returns:
            List of repository names in "owner/repo" format
        """
        search_query = f"{query} language:{language} stars:>={min_stars}"
        repos = self.github.search_repositories(query=search_query, sort="stars", order="desc")

        results = []
        for repo in repos[:max_results]:
            results.append(repo.full_name)

        return results

    def check_rate_limit(self) -> dict:
        """Check GitHub API rate limit status."""
        rate_limit = self.github.get_rate_limit()
        return {
            "core_remaining": rate_limit.core.remaining,
            "core_limit": rate_limit.core.limit,
            "core_reset": rate_limit.core.reset,
            "search_remaining": rate_limit.search.remaining,
            "search_limit": rate_limit.search.limit,
        }

