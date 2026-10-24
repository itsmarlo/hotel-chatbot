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
On 27 supported questions, including seven Beirut sightseeing questions, semantic retrieval
hits the expected source first on 24 (88.9%). It rejects both unsupported questions in the
small retrieval set. On macOS, setting `OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false` avoids
an intermittent local model crash during this run. The same corpus and questions
were authored for this project, so the result can be optimistic.

## End-to-end dialogue checks

Run `HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 python scripts/evaluate_dialogues.py`.
Thirty dialogues, ten each in English, German, and French, passed on 2026-10-19. Each language
contains two completed bookings, two unconfirmed quotes, two invalid date requests, two grounded
policy questions, and two unsupported questions. The script checks the database before
confirmation and after successful confirmation. All six completed bookings required the exact
confirmation phrase, and none of the other 24 dialogues created a reservation. The current
evaluator separately counts premature booking claims in the assistant reply and database
writes. A second, additional set of challenge prompts covers sightseeing phrasing, mixed
language, invalid dates, ambiguous confirmation, and changed checkout dates. Run
`python scripts/evaluate_challenges.py` to check it. These prompts were also authored for
the project and are not independently labeled guest traffic. The 11 additional prompts
passed locally. The 30 original dialogues passed in semantic mode with zero premature
booking claims and zero premature database writes.

## Qwen proposal and single-prompt comparison

The optional Qwen mode uses the same 30 authored dialogues. Its interpreter returns a
grammar-constrained JSON proposal; date and guest values are checked against the message, while
the booking workflow alone controls database writes. Policy questions route through the
retriever before a model proposal can redirect them. The baseline uses the same pinned Qwen
model with one prompt and no tools or database access. Its `booking_created` output is a model
claim, not a reservation.

To reproduce the raw results after downloading Qwen and installing `.[dev,rag,qwen]`:

```bash
HOTEL_NLP_MODE=qwen HOTEL_QWEN_MODEL_PATH=models/cache/qwen \
HOTEL_QWEN_LOCAL_ONLY=1 HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 \
python scripts/evaluate_dialogues.py > reports/qwen_guarded.json
HOTEL_QWEN_MODEL_PATH=models/cache/qwen HOTEL_QWEN_LOCAL_ONLY=1 \
python scripts/evaluate_baseline.py > reports/qwen_baseline.json
python scripts/record_qwen_comparison.py
```

The summary is in [`qwen_comparison.json`](qwen_comparison.json); raw outcomes are in `reports/`.
The first Qwen run passed 18/30. Its 12 routing failures were policy and unsupported
questions across the three languages. Routing verified questions before model interpretation
raised the rerun to 30/30, with zero database writes before confirmation. This is an
iterative regression check: the same authored questions informed the fix and the rerun.

The single-prompt baseline claimed a booking before confirmation in 24/30 cases, including
all six invalid-date cases. It claimed all six valid bookings after confirmation. The guarded
system created those six reservations only after exact confirmation, as verified in SQLite.
The current guarded semantic run measures both claim wording and database effects before
confirmation: 0/30 premature text claims and 0 database writes. The historical Qwen guarded
report measured writes but did not record wording. The Qwen baseline's `booking_created`
boolean is a different claim measure and has no database action to compare directly. These
authored results do not support a real-world relative-risk estimate.

The baseline criterion counts the model's boolean booking claim, even where its prose is
inconsistent. It therefore measures unsafe model intent, not whether a guest would believe the
reply or whether an actual reservation could be created. This comparison is descriptive for
the synthetic suite, and the baseline has no database action to compare directly.

## Next optimization cycle

Collect consented, anonymized, independently labeled guest utterances across the three
languages; expand the BIO labels; reserve a new untouched test set; measure entity-level F1 and
unsafe booking rate; inspect mistakes; then recalibrate retrieval and intent thresholds. The
current synthetic tests should remain regression checks, not a substitute for that evaluation.
