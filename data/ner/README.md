# Synthetic NER starter data

`starter.jsonl` contains manually authored **synthetic** token/tag examples in BIO format. Each
line has aligned `tokens` and `tags` arrays plus an `is_synthetic` marker. It exists to define
the annotation contract, not to support credible model-performance claims.

Phase C will expand and audit the annotations, define immutable splits, align subword labels,
train the token-classification model, and evaluate entity-level precision, recall, and F1.

