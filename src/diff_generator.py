"""Diff generation between repository versions."""

import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from git import Repo
from git.exc import GitCommandError
from rich.console import Console

from .repo_discovery import ReleaseInfo

console = Console()


@dataclass
class FileDiff:
    """Diff information for a single file."""

    file_path: str
    change_type: str  # "added", "modified", "deleted", "renamed"
    old_path: Optional[str] = None  # For renamed files
    diff_content: str = ""
    additions: int = 0
    deletions: int = 0


@dataclass
class VersionDiff:
    """Complete diff between two versions."""

    repo_name: str
    from_version: str
    to_version: str
    file_diffs: list[FileDiff]
    total_additions: int
    total_deletions: int
    files_added: int
    files_modified: int
    files_deleted: int

    def to_text(self, include_header: bool = True) -> str:
        """
        Convert the diff to a text representation.

        Args:
            include_header: Whether to include a header with version info

        Returns:
            Text representation of the complete diff
        """
        lines = []

        if include_header:
            lines.append(f"# Repository: {self.repo_name}")
            lines.append(f"# From Version: {self.from_version}")
            lines.append(f"# To Version: {self.to_version}")
            lines.append(f"# Files Changed: {len(self.file_diffs)}")
            lines.append(f"# Additions: +{self.total_additions}")
            lines.append(f"# Deletions: -{self.total_deletions}")
            lines.append("")
            lines.append("# Summary:")
            lines.append(f"#   Added: {self.files_added}")
            lines.append(f"#   Modified: {self.files_modified}")
            lines.append(f"#   Deleted: {self.files_deleted}")
            lines.append("")

        for file_diff in self.file_diffs:
            lines.append(f"{'=' * 80}")
            if file_diff.change_type == "renamed":
                lines.append(f"RENAMED: {file_diff.old_path} -> {file_diff.file_path}")
            else:
                lines.append(f"{file_diff.change_type.upper()}: {file_diff.file_path}")
            lines.append(
                f"(+{file_diff.additions}, -{file_diff.deletions})"
            )
            lines.append(f"{'=' * 80}")
            lines.append(file_diff.diff_content)
            lines.append("")

        return "\n".join(lines)

    def to_unified_diff(self) -> str:
        """Get the diff in standard unified diff format."""
        return "\n".join(fd.diff_content for fd in self.file_diffs)


class DiffGenerator:
    """Generates diffs between repository versions."""

    def __init__(
        self,
        context_lines: int = 3,
        max_diff_size: int = 10485760,  # 10MB
    ):
        """
        Initialize the diff generator.

        Args:
            context_lines: Number of context lines in diff output
            max_diff_size: Maximum diff size in bytes
        """
        self.context_lines = context_lines
        self.max_diff_size = max_diff_size

    def generate_diff(
        self,
        repo_path: Path,
        from_release: ReleaseInfo,
        to_release: ReleaseInfo,
        file_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> VersionDiff:
        """
        Generate a diff between two versions.

        Args:
            repo_path: Path to the repository
            from_release: Starting version
            to_release: Ending version
            file_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude

        Returns:
            VersionDiff containing all file changes
        """
        console.print(
            f"[cyan]Generating diff: {from_release.tag_name} -> {to_release.tag_name}[/cyan]"
        )

        repo = Repo(repo_path)

        # Get the diff using git
        try:
            diff_index = repo.commit(from_release.commit_sha).diff(
                repo.commit(to_release.commit_sha)
            )
        except GitCommandError as e:
            raise RuntimeError(f"Failed to generate diff: {e}") from e

        file_diffs = []
        total_additions = 0
        total_deletions = 0
        files_added = 0
        files_modified = 0
        files_deleted = 0

        for diff_item in diff_index:
            # Determine file path
            file_path = diff_item.b_path or diff_item.a_path

            # Apply file pattern filters
            if file_patterns and not self._matches_patterns(file_path, file_patterns):
                continue
            if exclude_patterns and self._matches_patterns(file_path, exclude_patterns):
                continue

            # Determine change type
            if diff_item.new_file:
                change_type = "added"
                files_added += 1
            elif diff_item.deleted_file:
                change_type = "deleted"
                files_deleted += 1
            elif diff_item.renamed:
                change_type = "renamed"
                files_modified += 1
            else:
                change_type = "modified"
                files_modified += 1

            # Generate the diff content
            diff_content, additions, deletions = self._get_diff_content(
                repo,
                from_release.commit_sha,
                to_release.commit_sha,
                diff_item,
            )

            file_diffs.append(
                FileDiff(
                    file_path=file_path,
                    change_type=change_type,
                    old_path=diff_item.a_path if diff_item.renamed else None,
                    diff_content=diff_content,
                    additions=additions,
                    deletions=deletions,
                )
            )

            total_additions += additions
            total_deletions += deletions

        # Sort diffs by file path
        file_diffs.sort(key=lambda d: d.file_path)

        console.print(
            f"[green]Generated diff with {len(file_diffs)} files changed "
            f"(+{total_additions}, -{total_deletions})[/green]"
        )

        return VersionDiff(
            repo_name=repo_path.name,
            from_version=from_release.tag_name,
            to_version=to_release.tag_name,
            file_diffs=file_diffs,
            total_additions=total_additions,
            total_deletions=total_deletions,
            files_added=files_added,
            files_modified=files_modified,
            files_deleted=files_deleted,
        )

    def _get_diff_content(
        self,
        repo: Repo,
        from_sha: str,
        to_sha: str,
        diff_item,
    ) -> tuple[str, int, int]:
        """Get the diff content for a single file."""
        additions = 0
        deletions = 0

        try:
            # Get file contents at both commits
            old_content = ""
            new_content = ""

            if not diff_item.new_file and diff_item.a_path:
                try:
                    old_blob = repo.commit(from_sha).tree / diff_item.a_path
                    old_content = old_blob.data_stream.read().decode("utf-8", errors="replace")
                except KeyError:
                    pass

            if not diff_item.deleted_file and diff_item.b_path:
                try:
                    new_blob = repo.commit(to_sha).tree / diff_item.b_path
                    new_content = new_blob.data_stream.read().decode("utf-8", errors="replace")
                except KeyError:
                    pass

            # Generate unified diff
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            diff_lines = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{diff_item.a_path or diff_item.b_path}",
                    tofile=f"b/{diff_item.b_path or diff_item.a_path}",
                    n=self.context_lines,
                )
            )

            # Count additions and deletions
            for line in diff_lines:
                if line.startswith("+") and not line.startswith("+++"):
                    additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    deletions += 1

            diff_content = "".join(diff_lines)

            # Truncate if too large
            if len(diff_content) > self.max_diff_size:
                diff_content = (
                    diff_content[: self.max_diff_size]
                    + f"\n... [truncated, diff too large] ..."
                )

            return diff_content, additions, deletions

        except Exception as e:
            return f"Error generating diff: {e}", 0, 0

    def _matches_patterns(self, path: str, patterns: list[str]) -> bool:
        """Check if a path matches any of the given patterns."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            # Also check just the filename
            if fnmatch.fnmatch(path.split("/")[-1], pattern):
                return True
        return False

    def generate_diff_from_git(
        self,
        repo_path: Path,
        from_ref: str,
        to_ref: str,
    ) -> str:
        """
        Generate a diff using git command line.

        This is faster for large diffs.

        Args:
            repo_path: Path to the repository
            from_ref: Starting reference (tag, commit, branch)
            to_ref: Ending reference

        Returns:
            Unified diff as string
        """
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "diff",
                    f"--unified={self.context_lines}",
                    from_ref,
                    to_ref,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to generate diff: {e.stderr}") from e

    def get_changed_files(
        self,
        repo_path: Path,
        from_ref: str,
        to_ref: str,
    ) -> list[dict]:
        """
        Get a list of changed files between two references.

        Args:
            repo_path: Path to the repository
            from_ref: Starting reference
            to_ref: Ending reference

        Returns:
            List of dicts with file info
        """
        repo = Repo(repo_path)

        try:
            diff_index = repo.commit(from_ref).diff(repo.commit(to_ref))
        except GitCommandError as e:
            raise RuntimeError(f"Failed to get changed files: {e}") from e

        files = []
        for diff_item in diff_index:
            files.append(
                {
                    "path": diff_item.b_path or diff_item.a_path,
                    "old_path": diff_item.a_path,
                    "change_type": (
                        "added"
                        if diff_item.new_file
                        else "deleted"
                        if diff_item.deleted_file
                        else "renamed"
                        if diff_item.renamed
                        else "modified"
                    ),
                }
            )

        return files

    def get_diff_stats(
        self,
        repo_path: Path,
        from_ref: str,
        to_ref: str,
    ) -> dict:
        """Get statistics about a diff."""
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "diff",
                    "--stat",
                    from_ref,
                    to_ref,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return {"stat_output": result.stdout}
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get diff stats: {e.stderr}") from e

