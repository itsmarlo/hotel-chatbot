# Development evaluation

All examples in this report are hand-authored synthetic prompts for the fictional hotel. They
check the code path and known edge cases; they do not estimate quality on real guest messages.

## Supervised language models

Run `python scripts/train_models.py` to retrain and write `models/trained/evaluation.json`.
The 102 intent examples comprise 66 training, 18 development, and 18 test examples. Feature
selection uses the development split. The chosen multilingual embedding classifier scored
15/18 correct on test (accuracy 0.833; macro F1 0.830). The two word/character baselines scored
lower on development macro F1 (0.771 and 0.716 versus 0.828 for the selected model).

The BIO tagger has only six training, two development, and two test utterances. Its test token
F1 excluding `O` is 0.632. This is too little evidence for entity-level performance. Token
proposals appear in API debug output and are not used to update reservation state.

## Policy retrieval

Run `HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 python scripts/evaluate_retrieval.py`.
On 20 supported questions, semantic retrieval hits the expected source first on 17 (85%). It
rejects both unsupported questions in the small retrieval set. The same corpus and questions
were authored for this project, so the result can be optimistic.

## End-to-end dialogue checks

Run `HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 python scripts/evaluate_dialogues.py`.
Thirty dialogues, ten each in English, German, and French, passed on 2026-10-19. Each language
contains two completed bookings, two unconfirmed quotes, two invalid date requests, two grounded
policy questions, and two unsupported questions. The script checks the database before
confirmation and after successful confirmation. All six completed bookings required the exact
confirmation phrase, and none of the other 24 dialogues created a reservation.

## Next optimization cycle

Collect consented, anonymized, independently labeled guest utterances across the three
languages; expand the BIO labels; reserve a new untouched test set; measure entity-level F1 and
unsafe booking rate; inspect mistakes; then recalibrate retrieval and intent thresholds. The
current synthetic tests should remain regression checks, not a substitute for that evaluation.
