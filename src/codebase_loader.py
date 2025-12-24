"""Codebase loading and serialization."""

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

console = Console()


@dataclass
class FileInfo:
    """Information about a single file."""

    path: str
    content: str
    size: int
    language: str


@dataclass
class CodebaseSnapshot:
    """A snapshot of a codebase at a specific version."""

    repo_name: str
    version: str
    files: list[FileInfo]
    total_size: int

    def to_text(self, include_header: bool = True) -> str:
        """
        Convert the codebase to a text representation.

        Args:
            include_header: Whether to include a header with repo info

        Returns:
            Text representation of the entire codebase
        """
        lines = []

        if include_header:
            lines.append(f"# Repository: {self.repo_name}")
            lines.append(f"# Version: {self.version}")
            lines.append(f"# Total Files: {len(self.files)}")
            lines.append(f"# Total Size: {self.total_size:,} bytes")
            lines.append("")

        for file_info in self.files:
            lines.append(f"{'=' * 80}")
            lines.append(f"FILE: {file_info.path}")
            lines.append(f"{'=' * 80}")
            lines.append(file_info.content)
            lines.append("")

        return "\n".join(lines)


# Language detection based on file extension
LANGUAGE_MAP = {
    ".py": "python",
    ".pyx": "cython",
    ".pxd": "cython",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".sql": "sql",
    ".proto": "protobuf",
    ".cmake": "cmake",
    ".make": "makefile",
    ".dockerfile": "dockerfile",
}


class CodebaseLoader:
    """Handles loading and serializing codebases."""

    def __init__(
        self,
        max_file_size: int = 1048576,  # 1MB
        max_codebase_size: int = 104857600,  # 100MB
    ):
        """
        Initialize the codebase loader.

        Args:
            max_file_size: Maximum size of individual files to include
            max_codebase_size: Maximum total size of codebase to include
        """
        self.max_file_size = max_file_size
        self.max_codebase_size = max_codebase_size

    def load_codebase(
        self,
        repo_path: Path,
        repo_name: str,
        version: str,
        file_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> CodebaseSnapshot:
        """
        Load a codebase from a directory.

        Args:
            repo_path: Path to the repository
            repo_name: Name of the repository
            version: Version/tag of the repository
            file_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude

        Returns:
            CodebaseSnapshot containing all matching files
        """
        if file_patterns is None:
            file_patterns = ["*.py", "*.cpp", "*.h", "*.cu", "*.cuh"]
        if exclude_patterns is None:
            exclude_patterns = ["tests/*", "test/*", "docs/*", "*_test.py", "test_*.py"]

        files = []
        total_size = 0

        # Collect all matching files
        all_files = list(self._find_matching_files(repo_path, file_patterns, exclude_patterns))

        console.print(f"[cyan]Loading {len(all_files)} files from {repo_name}@{version}...[/cyan]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Loading files...", total=len(all_files))

            for file_path in all_files:
                progress.update(task, advance=1)

                # Check if we've exceeded the max codebase size
                if total_size >= self.max_codebase_size:
                    console.print(
                        f"[yellow]Warning: Reached max codebase size "
                        f"({self.max_codebase_size:,} bytes)[/yellow]"
                    )
                    break

                file_info = self._load_file(repo_path, file_path)
                if file_info:
                    files.append(file_info)
                    total_size += file_info.size

        # Sort files by path for consistent ordering
        files.sort(key=lambda f: f.path)

        console.print(
            f"[green]Loaded {len(files)} files ({total_size:,} bytes) "
            f"from {repo_name}@{version}[/green]"
        )

        return CodebaseSnapshot(
            repo_name=repo_name,
            version=version,
            files=files,
            total_size=total_size,
        )

    def _find_matching_files(
        self,
        repo_path: Path,
        file_patterns: list[str],
        exclude_patterns: list[str],
    ):
        """Find all files matching the given patterns."""
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            rel_root = Path(root).relative_to(repo_path)

            for file in files:
                if file.startswith("."):
                    continue

                rel_path = str(rel_root / file)

                # Check if file matches any include pattern
                matches_include = any(
                    fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(rel_path, pattern)
                    for pattern in file_patterns
                )

                if not matches_include:
                    continue

                # Check if file matches any exclude pattern
                matches_exclude = any(
                    fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns
                )

                if matches_exclude:
                    continue

                yield Path(root) / file

    def _load_file(self, repo_path: Path, file_path: Path) -> Optional[FileInfo]:
        """Load a single file."""
        try:
            file_size = file_path.stat().st_size

            # Skip files that are too large
            if file_size > self.max_file_size:
                return None

            # Skip binary files
            if self._is_binary(file_path):
                return None

            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()

            rel_path = str(file_path.relative_to(repo_path))
            language = self._detect_language(file_path)

            return FileInfo(
                path=rel_path,
                content=content,
                size=file_size,
                language=language,
            )

        except (OSError, UnicodeDecodeError) as e:
            console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
            return None

    def _is_binary(self, file_path: Path) -> bool:
        """Check if a file is binary."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes which typically indicate binary
                if b"\x00" in chunk:
                    return True
                # Check if the content can be decoded as UTF-8
                try:
                    chunk.decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    return True
        except OSError:
            return True

    def _detect_language(self, file_path: Path) -> str:
        """Detect the programming language of a file."""
        ext = file_path.suffix.lower()
        return LANGUAGE_MAP.get(ext, "unknown")

    def merge_codebases(
        self,
        primary: CodebaseSnapshot,
        dependencies: list[CodebaseSnapshot],
    ) -> CodebaseSnapshot:
        """
        Merge a primary codebase with its dependencies.

        Args:
            primary: The main codebase
            dependencies: List of dependency codebases

        Returns:
            Merged codebase snapshot
        """
        all_files = list(primary.files)
        total_size = primary.total_size

        for dep in dependencies:
            # Prefix dependency files with the repo name
            for file_info in dep.files:
                prefixed_file = FileInfo(
                    path=f"__deps__/{dep.repo_name}/{file_info.path}",
                    content=file_info.content,
                    size=file_info.size,
                    language=file_info.language,
                )
                all_files.append(prefixed_file)
                total_size += file_info.size

        return CodebaseSnapshot(
            repo_name=primary.repo_name,
            version=primary.version,
            files=all_files,
            total_size=total_size,
        )

    def get_file_tree(self, snapshot: CodebaseSnapshot) -> str:
        """Generate a file tree representation of the codebase."""
        lines = [f"# File Tree for {snapshot.repo_name}@{snapshot.version}", ""]

        # Build tree structure
        tree: dict = {}
        for file_info in snapshot.files:
            parts = file_info.path.split("/")
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = None

        # Render tree
        def render_tree(node: dict, prefix: str = "") -> list[str]:
            result = []
            items = sorted(node.items())
            for i, (name, subtree) in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                result.append(f"{prefix}{connector}{name}")
                if subtree is not None:
                    extension = "    " if is_last else "│   "
                    result.extend(render_tree(subtree, prefix + extension))
            return result

        lines.extend(render_tree(tree))
        return "\n".join(lines)

