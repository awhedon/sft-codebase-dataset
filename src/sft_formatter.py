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
        streaming: bool = True,
    ):
        """
        Initialize the SFT formatter.

        Args:
            output_dir: Directory to save output files
            system_prompt: Custom system prompt for chat format
            compress: Whether to compress output files
            streaming: Whether to stream examples to disk incrementally
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.compress = compress
        self.streaming = streaming
        
        # For streaming mode
        self._stream_file = None
        self._stream_chat_file = None
        self._streamed_examples: list[SFTExample] = []  # Keep lightweight metadata only
    
    def start_streaming(self, filename: str = "dataset.jsonl") -> None:
        """Start streaming mode - open files for incremental writes."""
        if not self.streaming:
            return
            
        output_path = self.output_dir / filename
        chat_path = self.output_dir / filename.replace(".jsonl", "_chat.jsonl")
        
        if self.compress:
            output_path = output_path.with_suffix(".jsonl.gz")
            chat_path = chat_path.with_suffix(".jsonl.gz")
            self._stream_file = gzip.open(output_path, "wt", encoding="utf-8")
            self._stream_chat_file = gzip.open(chat_path, "wt", encoding="utf-8")
        else:
            self._stream_file = open(output_path, "w", encoding="utf-8")
            self._stream_chat_file = open(chat_path, "w", encoding="utf-8")
        
        self._streamed_examples = []
        console.print(f"[cyan]Streaming output to {output_path}[/cyan]")
    
    def stream_example(self, example: SFTExample) -> None:
        """Write a single example to disk immediately (streaming mode)."""
        if self._stream_file is None:
            self.start_streaming()
        
        # Write to main file
        self._stream_file.write(json.dumps(example.to_dict()) + "\n")
        self._stream_file.flush()
        
        # Write to chat file
        chat_data = example.to_chat_format(self.system_prompt)
        self._stream_chat_file.write(json.dumps(chat_data) + "\n")
        self._stream_chat_file.flush()
        
        # Keep lightweight copy for stats (without the huge text fields)
        lightweight = SFTExample(
            repo_name=example.repo_name,
            from_version=example.from_version,
            to_version=example.to_version,
            created_at=example.created_at,
            input_text="",  # Don't store the huge text
            output_text="",
            input_tokens_approx=example.input_tokens_approx,
            output_tokens_approx=example.output_tokens_approx,
            num_files_in_codebase=example.num_files_in_codebase,
            num_files_changed=example.num_files_changed,
            num_dependencies=example.num_dependencies,
        )
        # Store actual sizes for stats
        lightweight._input_size = len(example.input_text)
        lightweight._output_size = len(example.output_text)
        self._streamed_examples.append(lightweight)
    
    def finish_streaming(self) -> list[SFTExample]:
        """Close streaming files and return metadata for stats."""
        if self._stream_file:
            self._stream_file.close()
            self._stream_file = None
        if self._stream_chat_file:
            self._stream_chat_file.close()
            self._stream_chat_file = None
        
        console.print(f"[green]Streamed {len(self._streamed_examples)} examples to disk[/green]")
        return self._streamed_examples
    
    def compute_statistics_streaming(self, examples: list[SFTExample]) -> dict:
        """Compute statistics from streamed examples (using stored sizes)."""
        if not examples:
            return {}

        input_sizes = [getattr(e, '_input_size', 0) for e in examples]
        output_sizes = [getattr(e, '_output_size', 0) for e in examples]
        input_tokens = [e.input_tokens_approx for e in examples]
        output_tokens = [e.output_tokens_approx for e in examples]
        files_in_codebase = [e.num_files_in_codebase for e in examples]
        files_changed = [e.num_files_changed for e in examples]

        repos = set(e.repo_name for e in examples)
        repo_counts = {
            repo: len([e for e in examples if e.repo_name == repo])
            for repo in repos
        }

        return {
            "num_examples": len(examples),
            "num_repos": len(repos),
            "repo_counts": repo_counts,
            "input_size_mean": sum(input_sizes) / len(input_sizes) if input_sizes else 0,
            "input_size_min": min(input_sizes) if input_sizes else 0,
            "input_size_max": max(input_sizes) if input_sizes else 0,
            "input_size_total": sum(input_sizes),
            "output_size_mean": sum(output_sizes) / len(output_sizes) if output_sizes else 0,
            "output_size_min": min(output_sizes) if output_sizes else 0,
            "output_size_max": max(output_sizes) if output_sizes else 0,
            "output_size_total": sum(output_sizes),
            "input_tokens_mean": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
            "input_tokens_min": min(input_tokens) if input_tokens else 0,
            "input_tokens_max": max(input_tokens) if input_tokens else 0,
            "input_tokens_total": sum(input_tokens),
            "output_tokens_mean": sum(output_tokens) / len(output_tokens) if output_tokens else 0,
            "output_tokens_min": min(output_tokens) if output_tokens else 0,
            "output_tokens_max": max(output_tokens) if output_tokens else 0,
            "output_tokens_total": sum(output_tokens),
            "files_in_codebase_mean": sum(files_in_codebase) / len(files_in_codebase) if files_in_codebase else 0,
            "files_in_codebase_min": min(files_in_codebase) if files_in_codebase else 0,
            "files_in_codebase_max": max(files_in_codebase) if files_in_codebase else 0,
            "files_changed_mean": sum(files_changed) / len(files_changed) if files_changed else 0,
            "files_changed_min": min(files_changed) if files_changed else 0,
            "files_changed_max": max(files_changed) if files_changed else 0,
        }

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

        stats = self.compute_statistics(examples)
        return self._format_stats_report(stats)

    def compute_statistics(self, examples: list[SFTExample]) -> dict:
        """Compute detailed statistics for the dataset."""
        if not examples:
            return {}

        input_sizes = [len(e.input_text) for e in examples]
        output_sizes = [len(e.output_text) for e in examples]
        input_tokens = [e.input_tokens_approx for e in examples]
        output_tokens = [e.output_tokens_approx for e in examples]
        files_in_codebase = [e.num_files_in_codebase for e in examples]
        files_changed = [e.num_files_changed for e in examples]

        repos = set(e.repo_name for e in examples)
        repo_counts = {
            repo: len([e for e in examples if e.repo_name == repo])
            for repo in repos
        }

        return {
            "num_examples": len(examples),
            "num_repos": len(repos),
            "repo_counts": repo_counts,
            # Input sizes (characters)
            "input_size_mean": sum(input_sizes) / len(input_sizes),
            "input_size_min": min(input_sizes),
            "input_size_max": max(input_sizes),
            "input_size_total": sum(input_sizes),
            # Output sizes (characters)
            "output_size_mean": sum(output_sizes) / len(output_sizes),
            "output_size_min": min(output_sizes),
            "output_size_max": max(output_sizes),
            "output_size_total": sum(output_sizes),
            # Token estimates
            "input_tokens_mean": sum(input_tokens) / len(input_tokens),
            "input_tokens_min": min(input_tokens),
            "input_tokens_max": max(input_tokens),
            "input_tokens_total": sum(input_tokens),
            "output_tokens_mean": sum(output_tokens) / len(output_tokens),
            "output_tokens_min": min(output_tokens),
            "output_tokens_max": max(output_tokens),
            "output_tokens_total": sum(output_tokens),
            # File counts
            "files_in_codebase_mean": sum(files_in_codebase) / len(files_in_codebase),
            "files_in_codebase_min": min(files_in_codebase),
            "files_in_codebase_max": max(files_in_codebase),
            "files_changed_mean": sum(files_changed) / len(files_changed),
            "files_changed_min": min(files_changed),
            "files_changed_max": max(files_changed),
        }

    def _format_stats_report(self, stats: dict) -> str:
        """Format statistics as a markdown report."""
        if not stats:
            return "No examples in dataset."

        def fmt_size(size: float) -> str:
            """Format byte size to human readable."""
            if size >= 1_000_000_000:
                return f"{size / 1_000_000_000:.2f} GB"
            elif size >= 1_000_000:
                return f"{size / 1_000_000:.2f} MB"
            elif size >= 1_000:
                return f"{size / 1_000:.2f} KB"
            return f"{size:.0f} bytes"

        report = f"""# SFT Dataset Statistics

## Overview
| Metric | Value |
|--------|-------|
| Total Examples | {stats['num_examples']:,} |
| Unique Repositories | {stats['num_repos']} |
| Total Input Size | {fmt_size(stats['input_size_total'])} |
| Total Output Size | {fmt_size(stats['output_size_total'])} |
| Total Tokens (approx) | {stats['input_tokens_total'] + stats['output_tokens_total']:,} |

## Input Statistics (Codebase)
| Metric | Characters | Tokens (approx) |
|--------|------------|-----------------|
| Mean | {fmt_size(stats['input_size_mean'])} | {stats['input_tokens_mean']:,.0f} |
| Min | {fmt_size(stats['input_size_min'])} | {stats['input_tokens_min']:,} |
| Max | {fmt_size(stats['input_size_max'])} | {stats['input_tokens_max']:,} |

## Output Statistics (Diff)
| Metric | Characters | Tokens (approx) |
|--------|------------|-----------------|
| Mean | {fmt_size(stats['output_size_mean'])} | {stats['output_tokens_mean']:,.0f} |
| Min | {fmt_size(stats['output_size_min'])} | {stats['output_tokens_min']:,} |
| Max | {fmt_size(stats['output_size_max'])} | {stats['output_tokens_max']:,} |

## File Statistics
| Metric | Files in Codebase | Files Changed |
|--------|-------------------|---------------|
| Mean | {stats['files_in_codebase_mean']:,.0f} | {stats['files_changed_mean']:,.0f} |
| Min | {stats['files_in_codebase_min']:,} | {stats['files_changed_min']:,} |
| Max | {stats['files_in_codebase_max']:,} | {stats['files_changed_max']:,} |

## Repositories
"""
        for repo, count in sorted(stats['repo_counts'].items()):
            report += f"- **{repo}**: {count} examples\n"

        return report

    def generate_dataset_card(self, examples: list[SFTExample], dataset_name: str) -> str:
        """Generate a HuggingFace dataset card (README.md)."""
        stats = self.compute_statistics(examples)
        
        if not stats:
            return "# Empty Dataset\n\nNo examples generated."

        def fmt_size(size: float) -> str:
            if size >= 1_000_000_000:
                return f"{size / 1_000_000_000:.2f} GB"
            elif size >= 1_000_000:
                return f"{size / 1_000_000:.2f} MB"
            elif size >= 1_000:
                return f"{size / 1_000:.2f} KB"
            return f"{size:.0f} bytes"

        repos_list = "\n".join(f"  - {repo}" for repo in sorted(stats['repo_counts'].keys()))

        card = f"""---
license: mit
task_categories:
  - text-generation
language:
  - code
tags:
  - code
  - sft
  - fine-tuning
  - software-engineering
  - version-control
size_categories:
  - {self._get_size_category(stats['num_examples'])}
---

# {dataset_name}

SFT (Supervised Fine-Tuning) dataset for training models to predict code changes between software versions.

## Dataset Description

Each example contains:
- **Input**: Complete codebase at version N (including dependencies)
- **Output**: Full diff to version N+1

This enables training models to understand software evolution patterns and predict code changes.

## Statistics

| Metric | Value |
|--------|-------|
| Total Examples | {stats['num_examples']:,} |
| Repositories | {stats['num_repos']} |
| Total Size | {fmt_size(stats['input_size_total'] + stats['output_size_total'])} |

### Input (Codebase)
| Metric | Size | Tokens |
|--------|------|--------|
| Mean | {fmt_size(stats['input_size_mean'])} | ~{stats['input_tokens_mean']:,.0f} |
| Min | {fmt_size(stats['input_size_min'])} | ~{stats['input_tokens_min']:,} |
| Max | {fmt_size(stats['input_size_max'])} | ~{stats['input_tokens_max']:,} |

### Output (Diff)
| Metric | Size | Tokens |
|--------|------|--------|
| Mean | {fmt_size(stats['output_size_mean'])} | ~{stats['output_tokens_mean']:,.0f} |
| Min | {fmt_size(stats['output_size_min'])} | ~{stats['output_tokens_min']:,} |
| Max | {fmt_size(stats['output_size_max'])} | ~{stats['output_tokens_max']:,} |

## Source Repositories

{repos_list}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{dataset_name}")

# Access an example
example = dataset["train"][0]
print(f"Repo: {{example['repo_name']}}")
print(f"Version: {{example['from_version']}} -> {{example['to_version']}}")
print(f"Input length: {{len(example['input_text']):,}} chars")
print(f"Output length: {{len(example['output_text']):,}} chars")
```

## Data Format

Each example contains:
- `repo_name`: Repository name (e.g., "huggingface/transformers")
- `from_version`: Starting version tag
- `to_version`: Target version tag
- `input_text`: Complete codebase as text
- `output_text`: Unified diff between versions
- `input_tokens_approx`: Approximate input token count
- `output_tokens_approx`: Approximate output token count
- `num_files_in_codebase`: Number of files in input
- `num_files_changed`: Number of files in diff
- `num_dependencies`: Number of dependency codebases included
- `created_at`: Timestamp of example creation

## License

MIT License

## Citation

```bibtex
@dataset{{sft_codebase_diffs,
  title = {{SFT Codebase Diffs Dataset}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{dataset_name}}}
}}
```
"""
        return card

    def _get_size_category(self, num_examples: int) -> str:
        """Get HuggingFace size category."""
        if num_examples < 1000:
            return "n<1K"
        elif num_examples < 10000:
            return "1K<n<10K"
        elif num_examples < 100000:
            return "10K<n<100K"
        elif num_examples < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"

    def _format_dataset_card_from_stats(self, stats: dict, dataset_name: str) -> str:
        """Generate a HuggingFace dataset card from pre-computed stats."""
        if not stats:
            return "# Empty Dataset\n\nNo examples generated."

        def fmt_size(size: float) -> str:
            if size >= 1_000_000_000:
                return f"{size / 1_000_000_000:.2f} GB"
            elif size >= 1_000_000:
                return f"{size / 1_000_000:.2f} MB"
            elif size >= 1_000:
                return f"{size / 1_000:.2f} KB"
            return f"{size:.0f} bytes"

        repos_list = "\n".join(f"  - {repo}" for repo in sorted(stats.get('repo_counts', {}).keys()))

        card = f"""---
license: mit
task_categories:
  - text-generation
language:
  - code
tags:
  - code
  - sft
  - fine-tuning
  - software-engineering
  - version-control
size_categories:
  - {self._get_size_category(stats.get('num_examples', 0))}
---

# {dataset_name}

SFT (Supervised Fine-Tuning) dataset for training models to predict code changes between software versions.

## Dataset Description

Each example contains:
- **Input**: Complete codebase at version N (including dependencies)
- **Output**: Full diff to version N+1

This enables training models to understand software evolution patterns and predict code changes.

## Statistics

| Metric | Value |
|--------|-------|
| Total Examples | {stats.get('num_examples', 0):,} |
| Repositories | {stats.get('num_repos', 0)} |
| Total Size | {fmt_size(stats.get('input_size_total', 0) + stats.get('output_size_total', 0))} |

### Input (Codebase)
| Metric | Size | Tokens |
|--------|------|--------|
| Mean | {fmt_size(stats.get('input_size_mean', 0))} | ~{stats.get('input_tokens_mean', 0):,.0f} |
| Min | {fmt_size(stats.get('input_size_min', 0))} | ~{stats.get('input_tokens_min', 0):,} |
| Max | {fmt_size(stats.get('input_size_max', 0))} | ~{stats.get('input_tokens_max', 0):,} |

### Output (Diff)
| Metric | Size | Tokens |
|--------|------|--------|
| Mean | {fmt_size(stats.get('output_size_mean', 0))} | ~{stats.get('output_tokens_mean', 0):,.0f} |
| Min | {fmt_size(stats.get('output_size_min', 0))} | ~{stats.get('output_tokens_min', 0):,} |
| Max | {fmt_size(stats.get('output_size_max', 0))} | ~{stats.get('output_tokens_max', 0):,} |

## Source Repositories

{repos_list}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{dataset_name}")

# Access an example
example = dataset["train"][0]
print(f"Repo: {{example['repo_name']}}")
print(f"Version: {{example['from_version']}} -> {{example['to_version']}}")
print(f"Input length: {{len(example['input_text']):,}} chars")
print(f"Output length: {{len(example['output_text']):,}} chars")
```

## License

MIT License
"""
        return card

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

