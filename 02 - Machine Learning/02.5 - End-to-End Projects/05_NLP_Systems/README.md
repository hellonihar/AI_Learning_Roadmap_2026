# NLP Systems — End-to-End Projects

Classical NLP projects using scikit-learn. No transformers, no deep learning.

| Project | Dataset | Approach | Key Concept |
|---------|---------|----------|-------------|
| 1. Sentiment Analysis | IMDB Reviews | TF-IDF + LogisticRegression / MultinomialNB | Unigrams vs Bigrams |
| 2. Resume Screening | Synthetic | TF-IDF + OneVsRest SVM | Multi-label, pattern extraction |
| 3. News Categorization | 20 Newsgroups | TF-IDF + Linear SVC + LDA | Topic modeling |
| 4. Text Similarity / Duplicate Detection | Quora Duplicates | TF-IDF cosine + Jaccard + WMD | Distance metrics |

### Setup

```bash
pip install scikit-learn numpy pandas nltk gensim
python -m nltk.downloader stopwords wordnet
```

### How to Run

Each project is self-contained. Navigate to the project folder and run the main script or notebook.

### Learning Path

1. Start with sentiment analysis (binary, easy eval)
2. Move to resume screening (multi-label, structured output)
3. Try news categorization (multi-class + topic modeling)
4. Finish with text similarity (metric learning, no classifier)
