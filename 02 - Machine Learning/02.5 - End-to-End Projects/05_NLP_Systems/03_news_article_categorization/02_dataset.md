# News Article Categorisation — Dataset

## Source
[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) — ~20,000 documents evenly distributed across 20 newsgroups. Loaded via `sklearn.datasets.fetch_20newsgroups`.

## Size
| Split | Samples |
|-------|---------|
| Train | 11,314 |
| Test  | 7,532 |

## Categories (4 Supergroups)
- **Comp**: comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x
- **Rec**: rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
- **Sci**: sci.crypt, sci.electronics, sci.med, sci.space
- **Misc**: misc.forsale
- **Politics**: talk.politics.guns, talk.politics.mideast, talk.politics.misc
- **Religion**: talk.religion.misc, alt.atheism, soc.religion.christian

## Features
- **Vocabulary size**: ~100,000 unique tokens
- **TF-IDF features**: 10,000 top features (unigrams + bigrams)
- **Document length**: 50–10,000 words
- **Target**: 20 classes (integer 0–19)

## Challenges
- Some classes are very similar (e.g., ibm.pc.hardware vs mac.hardware — both discuss hardware)
- Cross-posting — same article can appear in multiple newsgroups (approximated as single-label here)
- Technical jargon differs by group (crypto vs medicine vocabularies)
