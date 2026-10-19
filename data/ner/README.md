# Synthetic NER starter data

`starter.jsonl` contains manually authored **synthetic** token/tag examples in BIO format. Each
line has aligned `tokens` and `tags` arrays plus an `is_synthetic` marker. It exists to define
the annotation contract, not to support credible model-performance claims.

The training script uses the first six examples for training, the next two for development and
the final two for test. It fits a lexical context classifier and reports token F1 excluding `O`.
Ten authored synthetic examples cannot establish entity-level or real-world quality. The model
only supplies debug proposals; the booking guard uses explicit validated input. More independent
multilingual annotations and entity-level evaluation are required before applying its proposals.

