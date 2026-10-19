# Hotel AI Chatbot

An academic prototype of Bellhob, a guarded hotel booking assistant for the fictional
Beit Zeitouna hotel in Beirut. The Phase 1 concept calls for multilingual dialogue,
deterministic booking rules, policy retrieval, and measured evaluation.

The app now answers questions from authored hotel policies and handles guarded room requests
through the UI and API. The fictional data and prices are for demonstration; no payment is taken.

## 1. Project goal

The final MVP will interpret natural-language booking requests, maintain booking state across
turns, answer hotel-policy questions from local documents, and create reservations only after
the user confirms a valid quote. An LLM will never be the source of truth for booking state.

## 2. Architecture

```text
Gradio UI
  -> Conversation Controller                         (implemented)
     -> Conservative date and slot parser            (implemented)
     -> Deterministic BookingState                    (implemented)
        -> Guarded booking workflow + SQLite         (implemented)
        -> Hotel-policy retrieval + sources          (implemented)
```

The Gradio UI and FastAPI endpoint use the same controller. Conversation state is held in a
bounded, expiring in-process store. Run one application worker; durable multi-worker sessions
require a shared session store before a wider deployment.

An earlier project specification introduced optional FastAPI, SQLite, and Docker foundations.
The API, SQLite store, and Docker packaging now support the interactive prototype.

## 3. Current status

Implemented:

- Python package and pinned base dependencies
- environment-backed configuration and structured logging
- deterministic booking state with four mandatory and two optional slots
- ordered missing-slot detection, correction support, atomic validation, reset, and 10-turn limit
- configurable booking policy in `configs/booking.json`
- synthetic intent starter covering all six intents
- synthetic BIO-format NER starter covering all six entity types
- Gradio shell with conversation, booking summary, new-conversation control, and demo panel
- root `app.py` and `requirements.txt` for Hugging Face Spaces
- unit, ML-data-contract, and integration tests

Booking and retrieval:

- `BookingWorkflow` enforces search, quote, explicit confirmation, then creation
- SQLite creation atomically rechecks occupancy and the quoted nightly price
- corrected booking details invalidate the earlier search, quote, and confirmation
- policy answers return authored wording and a source ID; unsupported questions are declined
- semantic retrieval uses multilingual Sentence-Transformers and FAISS when enabled

Not implemented yet:

- tokenizers, model training, model inference, or model artifacts
- intent or NER evaluation results
- constrained LLM interpretation and robust free-form multilingual slot extraction
- measured semantic retrieval results on a held-out question set
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

The repository also has two deliberately small, manually authored synthetic starters:

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

## 8. Policy retrieval

Policies live in `data/knowledge_base/policies.json`. The default local mode uses conservative
keyword retrieval and works with the base install. To enable multilingual vector retrieval:

```bash
python -m pip install -e '.[rag]'
HOTEL_RAG_MODE=semantic python app.py
```

The semantic mode loads a pinned multilingual Sentence-Transformers model and builds an in-memory
FAISS index over the policy text. Its first local run downloads model files. The Docker build
downloads the model once and runs with local files only. Answers return the source policy ID and
decline questions below the retrieval threshold. On the small, manually authored set in
`data/knowledge_base/evaluation.json`, semantic Hit@1 is 17/20 (85%) and both unsupported
questions are declined. This is a development check, not a real-traffic accuracy estimate.

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

Try `When is check-in?` or `I want to book a double room`. Dates must use `YYYY-MM-DD`.
The assistant asks for missing information, quotes an available synthetic room, and only creates
a reservation after the exact confirmation phrase shown in the chat.

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
HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 python scripts/evaluate_retrieval.py
```

Run the evaluation script to recompute the small retrieval figures above. The full scripted
dialogue comparison in the concept has not been run yet.

## 11. Hugging Face Spaces deployment

For a Gradio Space:

1. Create a new Space using the Gradio SDK and Python 3.11.
2. Push this repository, including root `app.py`, `requirements.txt`, `pyproject.toml`, `src/`,
   `app/`, `configs/`, and `data/`.
3. Spaces installs `requirements.txt`, which installs the local package and its pinned runtime
   dependencies.
4. The Space starts `app.py`; Gradio binds to the platform-provided port when configured through
   `HOTEL_UI_PORT`.

The Docker build fetches the pinned embedding model. Spaces using `requirements.txt` run the
lightweight keyword mode unless the `rag` extra is installed and `HOTEL_RAG_MODE=semantic` is set.

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
  rag/policies.py              Grounded policy retrieval
  services/chat.py             Conversation controller
  services/interpret.py        Conservative slot parser
tests/{unit,ml,integration}/   Automated checks by layer
```

See [docs/roadmap.md](docs/roadmap.md) for the ordered next phases.
