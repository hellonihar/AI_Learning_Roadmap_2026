# Resume Screening — Deployment Notes

## Inference Latency

| Step | Time per resume |
|------|----------------|
| Text cleaning | ~0.5 ms |
| Skill extraction | ~0.3 ms |
| TF-IDF transform | ~1.5 ms |
| OneVsRest SVM (5 classifiers) | ~0.5 ms |
| **Total** | **~2.8 ms** |

Processing 500 resumes takes ~1.4 seconds. Suitable for batch screening.

## Model Size

| Artifact | Size |
|----------|------|
| TF-IDF vectoriser | ~5 MB |
| 5× LinearSVC models | ~2 MB total |
| Skill patterns + ML binariser | ~0.1 MB |
| **Total** | **~7 MB** |

## Retraining Strategy

- Retrain monthly as job market evolves (new skills emerge, old ones fade)
- When precision on a class drops below 85%, check for new vocabulary in that domain
- Skill patterns: review and extend quarterly (e.g., add "GenAI", "LLM", "RAG" in 2025+)
- Cold-start for new categories: collect 200+ labelled samples before adding to the model

## Deployment

- Batch processing via a queue (RabbitMQ / SQS) — submit folder of PDFs, get CSV of predictions
- For real-time screening (single resume upload), latency is sub-10 ms
- Parse PDFs with `PyMuPDF` (fitz) or `pdfplumber` before feeding to the pipeline
- Cache TF-IDF vectoriser + model in memory (FastAPI app reloads daily)
