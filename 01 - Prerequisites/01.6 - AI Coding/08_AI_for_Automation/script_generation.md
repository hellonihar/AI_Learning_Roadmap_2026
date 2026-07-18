# Script Generation

Generate utility scripts for common automation tasks.

## Prompt

```
"Write a Python script that:
- Scans a directory for all .csv files
- Reads each one, prints filename and shape
- Concatenates all into one DataFrame
- Saves to a Parquet file
- Uses argparse for input/output paths
- Has informative error messages
- Uses pathlib"
```

## Example

```python
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*.csv")
    args = parser.parse_args()

    files = list(args.input_dir.glob(args.pattern))
    if not files:
        print(f"No files matching '{args.pattern}' in {args.input_dir}")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"  {f.name}: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(args.output)
    print(f"Saved {combined.shape} to {args.output}")

if __name__ == "__main__":
    main()
```

## Automation Prompts

- "Write a script to rename all files in a directory by date modified"
- "Write a script to convert all .ipynb files to .py scripts"
- "Write a script to check for TODO/FIXME markers in a codebase"
- "Write a script to split a large CSV into monthly partitions"
- "Write a script to download all models from Hugging Face listed in a text file"
