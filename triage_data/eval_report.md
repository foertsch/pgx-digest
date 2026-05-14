# LearnedTriage — training report

Embedding-based classifier (fastembed BGE-small + sklearn
LogisticRegression) for routing (PGxFinding, DrugRec) pairs.

- Train examples: **113**
- Held-out examples: **28**
- Embedding dim: **384**
- Train accuracy: **0.8673**
- Class distribution (train): `{'llm': 83, 'template': 14, 'skip': 16}`

## Held-out accuracy: **0.821**

| Class | n | Accuracy |
|---|---|---|
| llm | 16 | 0.812 |
| template | 9 | 0.889 |
| skip | 3 | 0.667 |
