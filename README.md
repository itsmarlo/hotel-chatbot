# Hotel AI Chatbot

An academic prototype of Bellhob, a guarded hotel booking assistant for the fictional
Beit Zeitouna hotel in Beirut. The Phase 1 concept calls for multilingual dialogue,
deterministic booking rules, policy retrieval, and measured evaluation.

The initial application shell and booking state are complete. The first guarded booking workflow
is implemented as a Python service. No model has been trained and no performance result is claimed.

## 1. Project goal

The final MVP will interpret natural-language booking requests, maintain booking state across
turns, answer hotel-policy questions from local documents, and create reservations only after
the user confirms a valid quote. An LLM will never be the source of truth for booking state.

## 2. Architecture

```text
Gradio UI
  -> Conversation Controller                         (planned)
     -> Constrained multilingual LLM interpretation  (planned)
     -> Deterministic BookingState                    (implemented)
        -> Guarded booking workflow + SQLite         (Python service implemented)
        -> Hotel-policy FAISS retrieval + sources    (planned)
```

The current UI calls a framework-neutral placeholder service and stores a separate
`BookingState` object in each Gradio browser session. The guarded workflow is not yet connected
to the UI; it can be exercised directly through its Python API.

An earlier project specification introduced optional FastAPI, SQLite, and Docker foundations.
They remain tested and isolated, but are not required by the revised Phase A Gradio MVP.

## 3. Phase A status

Implemented:

- Python package and pinned Phase A dependencies
- environment-backed configuration and structured logging
- deterministic booking state with four mandatory and two optional slots
- ordered missing-slot detection, correction support, atomic validation, reset, and 10-turn limit
- configurable booking policy in `configs/booking.json`
- synthetic intent starter covering all six intents
- synthetic BIO-format NER starter covering all six entity types
- Gradio shell with conversation, booking summary, new-conversation control, and demo panel
- root `app.py` and `requirements.txt` for Hugging Face Spaces
- unit, ML-data-contract, and integration tests

New Phase 2 foundation:

- `BookingWorkflow` enforces search, quote, explicit confirmation, then creation
- SQLite creation atomically rechecks occupancy and the quoted nightly price
- corrected booking details invalidate the earlier search, quote, and confirmation

Not implemented yet:

- tokenizers, model training, model inference, or model artifacts
- intent or NER evaluation results
- conversation controller and UI booking flow
- hotel documents, SentenceTransformer embeddings, FAISS, or grounded policy responses
- end-to-end evaluation reports

## 4. Installation

Python 3.11 is the deployment target.

```bash
cd /path/to/hotel-chatbot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The existing local environment in this repository is named `venv`, so it can instead be
activated with `source venv/bin/activate`.

## 5. Dataset

Phase A provides two deliberately small, manually authored synthetic starters:

- `data/intent/starter.csv`: text, one of six intent labels, and a synthetic-data marker
- `data/ner/starter.jsonl`: pre-tokenized text, aligned BIO tags, and a synthetic-data marker

These files exist to define executable data contracts. They are not representative of real
hotel traffic and cannot support credible real-world performance claims. Their adjacent README
files document provenance and limitations.

## 6. Model development

The Phase 1 concept selects Qwen2.5-1.5B-Instruct for constrained multilingual intent and slot
proposals. The existing intent starter remains an optional experiment. Inference commands will
be documented after that layer exists.

## 7. NER-model training

The BIO starter remains an optional experiment and is not the selected extraction path in the
Phase 1 concept.

## 8. RAG indexing and evaluation

Planned after guarded tools. Fictional hotel-policy Markdown documents will be embedded with a
lightweight SentenceTransformer model and indexed with FAISS. Retrieval will be independently
evaluated with Hit@k or Recall@k. Answers will cite sources and decline when supporting evidence
is unavailable.

## 9. Run locally

Start the Gradio application:

```bash
source venv/bin/activate
python app.py
```

Open <http://127.0.0.1:7860>. To select another port:

```bash
HOTEL_UI_PORT=7861 python app.py
```

The current response clearly identifies itself as a Phase A placeholder. The state panel and
reset behavior are functional; conversation integration follows retrieval and LLM development.

Optional earlier foundations:

```bash
python scripts/seed_database.py
uvicorn hotel_assistant.api.main:app --reload --port 8000
```

API documentation is then available at <http://127.0.0.1:8000/docs>.

## 10. Tests and quality checks

```bash
pytest
ruff check .
mypy
```

No metric in this repository is hand-written or fabricated. Later evaluation scripts will
calculate and store their outputs beneath `reports/`.

## 11. Hugging Face Spaces deployment

For a Gradio Space:

1. Create a new Space using the Gradio SDK and Python 3.11.
2. Push this repository, including root `app.py`, `requirements.txt`, `pyproject.toml`, `src/`,
   `app/`, `configs/`, and `data/`.
3. Spaces installs `requirements.txt`, which installs the local package and its pinned runtime
   dependencies.
4. The Space starts `app.py`; Gradio binds to the platform-provided port when configured through
   `HOTEL_UI_PORT`.

Model artifacts are intentionally absent. Later phases must either commit suitably
small artifacts or download a versioned Hugging Face model during startup with graceful failure
handling.

## 12. Repository layout

```text
app.py                         Hugging Face Spaces entry point
app/gradio_app.py              Gradio UI factory
configs/booking.json           Required slots and turn policy
data/intent/                   Synthetic intent starter and provenance
data/ner/                      Synthetic BIO starter and provenance
reports/                       Generated evaluation-output location
src/hotel_assistant/
  dialogue/state.py            Deterministic per-session booking state
  services/booking.py           Guarded booking sequence and quote
  ml/{intent,slots}/            Optional supervised ML experiments
  rag/                         Future FAISS retrieval components
  services/chat.py             Phase A application seam
tests/{unit,ml,integration}/   Automated checks by layer
```

See [docs/roadmap.md](docs/roadmap.md) for the ordered next phases.
