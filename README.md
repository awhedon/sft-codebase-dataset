# SFT Codebase Dataset Generator

Generate Supervised Fine-Tuning (SFT) datasets from open-source repository version diffs. This tool creates input-output pairs where:

- **Input**: The complete codebase at a specific version, plus the source code of key dependencies (at their latest versions)
- **Output**: The full diff to the next release version

This enables training models to predict software evolution patterns, code changes, and version upgrades.

## Features

- 🔍 **Repository Discovery**: Automatically fetch metadata and releases from GitHub
- 📦 **Dependency Resolution**: Extract and include Python dependencies with known GitHub repos
- 🔄 **Version Management**: Process consecutive release pairs for temporal training data
- 📊 **Multiple Output Formats**: JSONL, Parquet, and HuggingFace Hub upload
- 🎯 **Configurable Filtering**: Include/exclude files by pattern
- 💾 **Efficient Caching**: Partial git clones and local caching

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/sft-codebase-dataset.git
cd sft-codebase-dataset

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### GitHub Token (Recommended)

Set a GitHub token for higher rate limits:

```bash
export GITHUB_TOKEN=your_github_token_here
```

## Quick Start

### 1. Generate dataset from configured repositories

```bash
# Use default configuration
sft-dataset generate

# With custom config
sft-dataset -c configs/repos.yaml generate

# Without dependencies (faster)
sft-dataset generate --no-deps

# Output as Parquet
sft-dataset generate --format parquet
```

### 2. Process a single repository

```bash
# Process HuggingFace transformers
sft-dataset single huggingface/transformers --max-versions 10

# Process vLLM
sft-dataset single vllm-project/vllm --max-versions 5
```

### 3. Explore repositories

```bash
# List configured repositories
sft-dataset list-repos

# View releases for a repository
sft-dataset releases pytorch/pytorch

# Search for ML repositories
sft-dataset search "machine learning" --min-stars 5000

# Check GitHub API rate limits
sft-dataset rate-limit
```

## Configuration

Edit `configs/repos.yaml` to customize which repositories to process:

```yaml
repositories:
  - name: "huggingface/transformers"
    description: "HuggingFace Transformers library"
    file_patterns:
      - "*.py"
    exclude_patterns:
      - "tests/*"
      - "docs/*"
    max_versions: 20  # Limit number of version pairs

  - name: "NVIDIA/TensorRT-LLM"
    description: "TensorRT-LLM for optimized inference"
    file_patterns:
      - "*.py"
      - "*.cpp"
      - "*.cu"

settings:
  max_file_size: 1048576  # 1MB per file
  max_codebase_size: 104857600  # 100MB total
  include_dependencies: true
  dependency_depth: 1
  output_format: "jsonl"
```

## Output Format

### JSONL Format

Each line contains a JSON object:

```json
{
  "repo_name": "huggingface/transformers",
  "from_version": "v4.35.0",
  "to_version": "v4.36.0",
  "input_text": "<repository>...",
  "output_text": "# Changes from v4.35.0 to v4.36.0\n\n--- a/src/...",
  "input_tokens_approx": 500000,
  "output_tokens_approx": 15000,
  "num_files_in_codebase": 1250,
  "num_files_changed": 87,
  "num_dependencies": 5
}
```

### Chat Format JSONL

For instruction-tuning compatible format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert software engineer..."},
    {"role": "user", "content": "<repository>..."},
    {"role": "assistant", "content": "# Changes from v4.35.0 to v4.36.0..."}
  ],
  "metadata": {...}
}
```

## Architecture

```
sft-codebase-dataset/
├── src/
│   ├── main.py              # CLI and orchestration
│   ├── config.py            # Configuration management
│   ├── repo_discovery.py    # GitHub API interactions
│   ├── version_fetcher.py   # Git operations
│   ├── codebase_loader.py   # File loading and serialization
│   ├── dependency_resolver.py # Dependency extraction
│   ├── diff_generator.py    # Diff generation
│   └── sft_formatter.py     # Output formatting
├── configs/
│   └── repos.yaml           # Repository configuration
└── output/                   # Generated datasets
```

## Supported Repositories

The tool works with any public GitHub repository that has tagged releases. Pre-configured repositories include:

| Repository | Description |
|------------|-------------|
| huggingface/transformers | HuggingFace Transformers |
| NVIDIA/TensorRT-LLM | TensorRT-LLM inference |
| vllm-project/vllm | vLLM inference engine |
| pytorch/pytorch | PyTorch framework |
| openai/triton | Triton compiler |
| langchain-ai/langchain | LangChain framework |

## Dependency Resolution

The tool automatically resolves Python dependencies from:
- `pyproject.toml`
- `setup.py`
- `requirements.txt`

Known packages are mapped to their GitHub repositories for source code inclusion. See `src/dependency_resolver.py` for the full mapping.

## Usage Examples

### Training Data for Code Evolution Models

```python
from datasets import load_dataset

# Load the generated dataset
dataset = load_dataset("json", data_files="output/dataset.jsonl.gz")

# Example: Filter by repository
transformers_data = dataset.filter(
    lambda x: "transformers" in x["repo_name"]
)

# Example: Filter by size
manageable_data = dataset.filter(
    lambda x: x["input_tokens_approx"] < 100000
)
```

### Upload to HuggingFace Hub

```bash
# Generate and upload
sft-dataset generate --upload

# Or upload manually
from datasets import Dataset
dataset = Dataset.from_json("output/dataset.jsonl.gz")
dataset.push_to_hub("your-username/sft-codebase-diffs")
```

## Resource Requirements

- **Disk Space**: 10-50GB for caching repositories
- **Memory**: 8GB+ recommended for large codebases
- **Network**: GitHub API access (authenticated recommended)
- **Time**: 1-10 minutes per repository depending on size

## Limitations

- Only processes tagged releases (not arbitrary commits)
- Dependency resolution limited to Python packages
- Large diffs may be truncated
- Binary files are excluded

## Contributing

Contributions welcome! Areas of interest:

1. Support for additional languages (Go, Rust, Java)
2. More sophisticated dependency resolution
3. Commit-level granularity option
4. Semantic diff generation
5. Performance optimizations

## License

MIT License - see LICENSE file for details.

## Citation

If you use this tool in research, please cite:

```bibtex
@software{sft_codebase_dataset,
  title = {SFT Codebase Dataset Generator},
  year = {2024},
  url = {https://github.com/your-org/sft-codebase-dataset}
}
```

