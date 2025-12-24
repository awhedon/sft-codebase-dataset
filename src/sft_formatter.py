"""SFT dataset formatting and export."""

import gzip
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

console = Console()


@dataclass
class SFTExample:
    """A single SFT training example."""

    # Metadata
    repo_name: str
    from_version: str
    to_version: str
    created_at: str

    # Input: Full codebase + dependencies
    input_text: str

    # Output: Full diff to next version
    output_text: str

    # Statistics
    input_tokens_approx: int
    output_tokens_approx: int
    num_files_in_codebase: int
    num_files_changed: int
    num_dependencies: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_chat_format(self, system_prompt: Optional[str] = None) -> dict:
        """
        Convert to chat format for instruction tuning.

        Returns:
            Dict with 'messages' key containing chat-format messages
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": self.input_text})
        messages.append({"role": "assistant", "content": self.output_text})

        return {
            "messages": messages,
            "metadata": {
                "repo_name": self.repo_name,
                "from_version": self.from_version,
                "to_version": self.to_version,
                "input_tokens_approx": self.input_tokens_approx,
                "output_tokens_approx": self.output_tokens_approx,
            },
        }


# Default system prompt for SFT
DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer. Given a complete codebase (including dependencies), predict the changes that will be made in the next version release. Output the changes as a unified diff format.

The input contains:
1. The complete source code of the main repository
2. Source code of key dependencies (under __deps__/)

Your task is to predict what changes, improvements, or bug fixes will be made to transform this codebase into the next release version."""


class SFTFormatter:
    """Handles formatting and exporting SFT datasets."""

    def __init__(
        self,
        output_dir: Path,
        system_prompt: Optional[str] = None,
        compress: bool = True,
    ):
        """
        Initialize the SFT formatter.

        Args:
            output_dir: Directory to save output files
            system_prompt: Custom system prompt for chat format
            compress: Whether to compress output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.compress = compress

    def create_example(
        self,
        repo_name: str,
        from_version: str,
        to_version: str,
        codebase_text: str,
        diff_text: str,
        num_files_in_codebase: int,
        num_files_changed: int,
        num_dependencies: int = 0,
    ) -> SFTExample:
        """
        Create an SFT example.

        Args:
            repo_name: Repository name
            from_version: Starting version
            to_version: Target version
            codebase_text: Full codebase as text
            diff_text: Full diff as text
            num_files_in_codebase: Number of files in codebase
            num_files_changed: Number of files changed in diff
            num_dependencies: Number of dependency codebases included

        Returns:
            SFTExample instance
        """
        # Create formatted input
        input_text = self._format_input(
            repo_name=repo_name,
            version=from_version,
            codebase_text=codebase_text,
        )

        # Create formatted output
        output_text = self._format_output(
            repo_name=repo_name,
            from_version=from_version,
            to_version=to_version,
            diff_text=diff_text,
        )

        # Approximate token counts (rough estimate: 4 chars per token)
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4

        return SFTExample(
            repo_name=repo_name,
            from_version=from_version,
            to_version=to_version,
            created_at=datetime.utcnow().isoformat(),
            input_text=input_text,
            output_text=output_text,
            input_tokens_approx=input_tokens,
            output_tokens_approx=output_tokens,
            num_files_in_codebase=num_files_in_codebase,
            num_files_changed=num_files_changed,
            num_dependencies=num_dependencies,
        )

    def _format_input(
        self,
        repo_name: str,
        version: str,
        codebase_text: str,
    ) -> str:
        """Format the input for SFT."""
        lines = [
            f"<repository>",
            f"<name>{repo_name}</name>",
            f"<version>{version}</version>",
            f"<codebase>",
            codebase_text,
            f"</codebase>",
            f"</repository>",
            "",
            "Based on the codebase above, predict the changes that will be made in the next version release. Output the changes as a unified diff.",
        ]
        return "\n".join(lines)

    def _format_output(
        self,
        repo_name: str,
        from_version: str,
        to_version: str,
        diff_text: str,
    ) -> str:
        """Format the output for SFT."""
        lines = [
            f"# Changes from {from_version} to {to_version}",
            "",
            diff_text,
        ]
        return "\n".join(lines)

    def save_examples_jsonl(
        self,
        examples: list[SFTExample],
        filename: str = "dataset.jsonl",
    ) -> Path:
        """
        Save examples to JSONL format.

        Args:
            examples: List of SFT examples
            filename: Output filename

        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / filename
        if self.compress:
            output_path = output_path.with_suffix(".jsonl.gz")

        console.print(f"[cyan]Saving {len(examples)} examples to {output_path}...[/cyan]")

        open_func = gzip.open if self.compress else open
        mode = "wt" if self.compress else "w"

        with open_func(output_path, mode, encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        console.print(f"[green]Saved to {output_path}[/green]")
        return output_path

    def save_examples_chat_jsonl(
        self,
        examples: list[SFTExample],
        filename: str = "dataset_chat.jsonl",
    ) -> Path:
        """
        Save examples to chat format JSONL.

        Args:
            examples: List of SFT examples
            filename: Output filename

        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / filename
        if self.compress:
            output_path = output_path.with_suffix(".jsonl.gz")

        console.print(f"[cyan]Saving {len(examples)} examples in chat format to {output_path}...[/cyan]")

        open_func = gzip.open if self.compress else open
        mode = "wt" if self.compress else "w"

        with open_func(output_path, mode, encoding="utf-8") as f:
            for example in examples:
                chat_data = example.to_chat_format(self.system_prompt)
                f.write(json.dumps(chat_data) + "\n")

        console.print(f"[green]Saved to {output_path}[/green]")
        return output_path

    def save_examples_parquet(
        self,
        examples: list[SFTExample],
        filename: str = "dataset.parquet",
    ) -> Path:
        """
        Save examples to Parquet format.

        Requires pyarrow or fastparquet.

        Args:
            examples: List of SFT examples
            filename: Output filename

        Returns:
            Path to the saved file
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            console.print("[red]pyarrow not installed, falling back to JSONL[/red]")
            return self.save_examples_jsonl(examples, filename.replace(".parquet", ".jsonl"))

        output_path = self.output_dir / filename

        console.print(f"[cyan]Saving {len(examples)} examples to {output_path}...[/cyan]")

        # Convert to columnar format
        data = {
            "repo_name": [e.repo_name for e in examples],
            "from_version": [e.from_version for e in examples],
            "to_version": [e.to_version for e in examples],
            "created_at": [e.created_at for e in examples],
            "input_text": [e.input_text for e in examples],
            "output_text": [e.output_text for e in examples],
            "input_tokens_approx": [e.input_tokens_approx for e in examples],
            "output_tokens_approx": [e.output_tokens_approx for e in examples],
            "num_files_in_codebase": [e.num_files_in_codebase for e in examples],
            "num_files_changed": [e.num_files_changed for e in examples],
            "num_dependencies": [e.num_dependencies for e in examples],
        }

        table = pa.Table.from_pydict(data)
        pq.write_table(table, output_path, compression="snappy")

        console.print(f"[green]Saved to {output_path}[/green]")
        return output_path

    def upload_to_hub(
        self,
        examples: list[SFTExample],
        dataset_name: str,
        private: bool = True,
    ) -> str:
        """
        Upload dataset to HuggingFace Hub.

        Args:
            examples: List of SFT examples
            dataset_name: Name for the dataset on the Hub
            private: Whether to make the dataset private

        Returns:
            URL of the uploaded dataset
        """
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
        except ImportError:
            raise RuntimeError("datasets and huggingface_hub required for Hub upload")

        console.print(f"[cyan]Uploading to HuggingFace Hub as {dataset_name}...[/cyan]")

        # Convert to HF Dataset
        data = [e.to_dict() for e in examples]
        dataset = Dataset.from_list(data)

        # Push to Hub
        dataset.push_to_hub(dataset_name, private=private)

        console.print(f"[green]Uploaded to https://huggingface.co/datasets/{dataset_name}[/green]")
        return f"https://huggingface.co/datasets/{dataset_name}"

    def generate_stats_report(
        self,
        examples: list[SFTExample],
    ) -> str:
        """Generate a statistics report for the dataset."""
        if not examples:
            return "No examples in dataset."

        total_input_tokens = sum(e.input_tokens_approx for e in examples)
        total_output_tokens = sum(e.output_tokens_approx for e in examples)
        total_files = sum(e.num_files_in_codebase for e in examples)
        total_changed = sum(e.num_files_changed for e in examples)

        repos = set(e.repo_name for e in examples)

        report = f"""
# SFT Dataset Statistics

## Overview
- Total Examples: {len(examples)}
- Unique Repositories: {len(repos)}
- Total Input Tokens (approx): {total_input_tokens:,}
- Total Output Tokens (approx): {total_output_tokens:,}

## Averages per Example
- Avg Input Tokens: {total_input_tokens // len(examples):,}
- Avg Output Tokens: {total_output_tokens // len(examples):,}
- Avg Files in Codebase: {total_files // len(examples):,}
- Avg Files Changed: {total_changed // len(examples):,}

## Repositories
"""
        for repo in sorted(repos):
            repo_examples = [e for e in examples if e.repo_name == repo]
            report += f"- {repo}: {len(repo_examples)} examples\n"

        return report

    def split_dataset(
        self,
        examples: list[SFTExample],
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> tuple[list[SFTExample], list[SFTExample]]:
        """
        Split examples into train and validation sets.

        Args:
            examples: List of examples to split
            train_ratio: Ratio of examples for training
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_examples, val_examples)
        """
        import random

        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

