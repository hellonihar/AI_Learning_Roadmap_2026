# 05.2.1 Data Versioning

## Why Data Versioning Matters

Machine learning models are defined as much by their training data as by their code. A model trained on dataset `v1` will behave differently from one trained on `v2`. Without data versioning, you cannot:

- **Reproduce** a past experiment (data changes, files are overwritten, old versions vanish).
- **Roll back** when a new data source introduces corruption or bias.
- **Collaborate** across a team — everyone needs the same dataset to train the same model.
- **Audit** which data produced which model for compliance (HIPAA, GDPR, SOX).

Data versioning solves these problems by treating datasets as first-class artifacts alongside source code, with the same semantics of commit, tag, branch, and diff.

## DVC (Data Version Control)

DVC is the most popular open-source data versioning tool for ML. It extends Git to handle large files and datasets. Key concepts:

### 1. Cache and Remote Storage

DVC never stores large data files directly in Git. Instead, it stores a **pointer file** (`.dvc`) that contains a hash (MD5) identifying the file content. The actual data lives in a **cache** (local `.dvc/cache`) and optionally in **remote storage** (S3, GCS, Azure Blob, HDFS, or a network drive).

```bash
dvc remote add myremote s3://my-bucket/dvcstore
dvc push   # upload cached files to remote
dvc pull   # download files from remote
```

### 2. `.dvc` Files

When you run `dvc add data/train.csv`:

- DVC moves the file to the cache (hardlink/symlink/copy depending on config).
- Creates `data/train.csv.dvc` — a small YAML or text file recording the hash, size, and output path.
- Adds `data/train.csv` to `.gitignore`.
- You commit the `.dvc` file to Git.

```yaml
# data/train.csv.dvc
outs:
  - md5: a1b2c3d4e5f6...
    size: 345678
    path: train.csv
```

Now `git checkout` of an older commit restores the old `.dvc` files, and `dvc checkout` restores the corresponding data files from cache.

### 3. DVC Pipelines

DVC goes beyond simple versioning — it models **data pipelines** as directed acyclic graphs (DAGs). Each stage reads dependencies and produces outputs. DVC tracks which files were inputs and which files were outputs, then recomputes only the stages whose dependencies changed.

```bash
dvc stage add -n featurize \
  -d data/raw.csv -d src/features.py \
  -o data/features.pkl \
  python src/features.py

dvc stage add -n train \
  -d data/features.pkl -d src/train.py \
  -o models/model.pkl \
  python src/train.py
```

Running `dvc repro` executes the DAG from the last changed stage onward. This guarantees **deterministic, cached reproduction** — if nothing changed, nothing runs.

### 4. Metrics and Parameters

DVC also tracks metrics (numbers from evaluation) and parameters (hyperparams):

```bash
dvc metrics show
dvc params diff
```

This gives a full lineage: which code, data, and parameters produced a given metric set.

## Storing Data in S3 / GCS

Production setups almost always use a remote store:

| Store      | Config                                    |
| ---------- | ----------------------------------------- |
| Amazon S3  | `dvc remote add -d s3://bucket/path`     |
| Google GCS | `dvc remote add -d gs://bucket/path`     |
| Azure      | `dvc remote add -d azure://container/path` |
| MinIO      | `dvc remote add -d s3://bucket/endpoint`  |

Credentials come from environment variables (`AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`) or cloud-specific config.

## Versioning Datasets Alongside Code

The typical workflow:

1. `git add src/` and `git commit -m "Add featurization code"`
2. `dvc add data/raw/` → creates `data/raw.dvc`
3. `git add data/raw.dvc` and `git commit -m "Add raw dataset v3"`
4. `dvc push` → uploads data to S3
5. Later: `git checkout <sha>` + `dvc checkout` → full restoration.

Tag a release: `git tag -a v1.0 -m "Release 1.0" && dvc tag v1.0`

## DVC vs Alternatives

| Tool         | Storage           | Pipeline | Notes                        |
| ------------ | ----------------- | -------- | ---------------------------- |
| DVC          | S3/GCS/any        | Yes      | Open source, Git-native      |
| Dolt         | SQL-based         | No       | Versioned SQL database       |
| Hugging Face Datasets | HF Hub    | No       | Dataset library + hub        |
| Quilt        | S3                | No       | Python data package manager  |
| LakeFS       | S3-compatible     | No       | Git-like branches on object store |

DVC remains the most widely adopted because it integrates naturally with existing Git workflows and adds pipeline orchestration at no extra cost.

## Best Practices

- **Keep raw data immutable** — never edit raw datasets in place. Create derived versions.
- **Use `.dvcignore`** like `.gitignore` to exclude ephemeral files.
- **Pin DVC versions** in CI to avoid hash-algorithm changes.
- **Push before sharing** — teammates should `dvc pull` after `git pull`.
- **For large directories** (thousands of files), use `dvc add --no-commit` then incremental push.

## Summary

Data versioning is the foundation of reproducible ML. DVC provides a Git-native, storage-agnostic way to version datasets, track pipeline stages, and collaborate with a team. By treating data as a tracked dependency, you eliminate the "works on my machine" problem that plagues ML projects.
