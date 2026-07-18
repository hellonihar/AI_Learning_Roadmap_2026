# Converting Scripts into Packages

Turn a single script into a proper installable Python package.

## Single Script → Package Structure

```
# Before:
my_script.py          # all code in one file

# After:
my_project/
├── pyproject.toml    # package metadata & dependencies
├── README.md
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── cli.py         # entry point
│       ├── core.py        # main logic
│       ├── utils.py       # helpers
│       └── config.py      # settings
├── tests/
│   ├── test_core.py
│   └── test_utils.py
└── .env.example
```

**Prompt:** "Convert this script into a Python package with pyproject.toml, entry points, and a CLI using argparse or click"

## pyproject.toml

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "click>=8.0",
]

[project.scripts]
my-project = "my_project.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

## CLI Entry Point

```python
# src/my_project/cli.py
import click

@click.command()
@click.option("--input", "-i", required=True, help="Input file path")
@click.option("--output", "-o", default="output.json", help="Output path")
def main(input: str, output: str):
    """Process data from INPUT and save results to OUTPUT."""
    ...

if __name__ == "__main__":
    main()
```
