# Resume Screening — Dataset

## Source
Synthetic dataset generated from a template system. 5,000 resumes covering 5 job categories:

0. Data Science
1. Software Engineering
2. Marketing
3. Finance
4. Human Resources

Each resume is 150–400 words of structured text (education, experience, skills sections).

## Size
| Split | Samples |
|-------|---------|
| Train | 4,000 |
| Test  | 1,000 |

## Features
- **Vocabulary size**: ~8,000 unique tokens
- **TF-IDF features**: 5,000 max features (unigrams + bigrams)
- **Skill extraction**: regex patterns over a curated skill list (Python, Excel, SEO, GAAP, etc.)
- **Target**: up to 2 labels per resume (multi-label; e.g., "Data Science" + "Software Engineering")
- **Multi-label binariser**: 5 binary columns

## Challenges
- Multi-label: a "Data Scientist" resume often qualifies for "Software Engineering"
- Sparse labels — some categories (HR) have fewer samples
- Skill extraction must handle synonyms ("ML", "Machine Learning", "machine-learning")
- Resumes use varied formatting (bullets, tables, paragraphs)
