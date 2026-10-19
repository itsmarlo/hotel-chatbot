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
     -> Trained multilingual intent and BIO models   (implemented in semantic mode)
     -> Conservative date and slot parser            (booking authority)
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
- fixed-split multilingual intent examples, reproducible training, and versioned model artifacts
- Gradio shell with conversation, booking summary, new-conversation control, and demo panel
- root `app.py` and `requirements.txt` for Hugging Face Spaces
- unit, ML-data-contract, and integration tests

Booking and retrieval:

- `BookingWorkflow` enforces search, quote, explicit confirmation, then creation
- SQLite creation atomically rechecks occupancy and the quoted nightly price
- corrected booking details invalidate the earlier search, quote, and confirmation
- policy answers return authored wording and a source ID; unsupported questions are declined
- semantic retrieval uses multilingual Sentence-Transformers and FAISS when enabled

Still needed before a real hotel launch:

- verified hotel policies, currency, taxes, payment and reservation operations
- durable shared sessions, authentication and customer-data controls
- realistic guest-language evaluations and calibrated confidence thresholds
- the concept's Qwen constrained-JSON interpreter and 30-dialogue baseline comparison

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

`data/intent/collected.csv` adds 72 hand-authored English, German and French messages with fixed
train/dev/test assignments. All data is synthetic and cannot support real-world performance
claims. The adjacent README files document provenance and limitations.

## 6. Model development

The supervised intent baseline compares word, character and multilingual embedding features on
the dev split, then evaluates the selected model on the fixed test split. The embedding model is
the same pinned multilingual Sentence-Transformers model used for RAG. Train both models with:

```bash
python -m pip install -e '.[dev,ml]'
python scripts/train_models.py
```

Artifacts and a machine-readable evaluation report are committed under `models/trained/`. The
selected intent model scored 15/18 test examples (83% accuracy; macro F1 0.83). This tiny,
authored test is an integration check, not a production accuracy estimate. The concept's Qwen
interpreter is a separate future component; this baseline does not imply Qwen was fine-tuned.

## 7. NER-model training

The BIO starter trains a token classifier with lexical and neighboring-token features. A small
dev split selects its regularization strength. Its current test token F1 excluding `O` is
reported in `models/trained/evaluation.json` on only two synthetic utterances. The model returns
entity proposals in debug output; validated booking details come from explicit date and slot
parsing. This avoids letting uncertain token tags change reservation state.

## 8. Policy retrieval

Policies live in `data/knowledge_base/policies.json`. The default local mode uses conservative
keyword retrieval and works with the base install. To enable multilingual vector retrieval:

```bash
python -m pip install -e '.[rag]'
HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 python app.py
```

The semantic mode loads a pinned multilingual Sentence-Transformers model and builds an in-memory
FAISS index over the policy text. Its first local run downloads model files. The Docker build
downloads the model once and runs with local files only. Answers return the source policy ID and
decline questions below the retrieval threshold. On the small, manually authored set in
`data/knowledge_base/evaluation.json`, semantic Hit@1 is 17/20 (85%) and both unsupported
questions are declined. This is a development check, not a real-traffic accuracy estimate.
In semantic mode the same encoder also runs the trained intent classifier and advisory BIO
tagger. The keyword mode works with the base install and skips those models.

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
python scripts/train_models.py
```

Run the scripts to recompute the small retrieval and training figures above. The full scripted
dialogue check runs with:

```bash
HOTEL_RAG_MODE=semantic HOTEL_RAG_LOCAL_ONLY=1 python scripts/evaluate_dialogues.py
```

It passed 30/30 authored scenarios in three languages. See [docs/evaluation.md](docs/evaluation.md)
for the method and limits. The concept's single-prompt baseline comparison remains to be built.

## 11. Hugging Face Spaces deployment

For a Gradio Space:

1. Create a new Space using the Gradio SDK and Python 3.11.
2. Push this repository, including root `app.py`, `requirements.txt`, `pyproject.toml`, `src/`,
   `app/`, `configs/`, `data/`, and `models/trained/`.
3. Spaces installs `requirements.txt`, which installs the local package and its pinned runtime
   dependencies.
4. The Space starts `app.py`; Gradio binds to the platform-provided port when configured through
   `HOTEL_UI_PORT`.

The Docker build fetches the pinned embedding model and includes the trained artifacts. Spaces
using `requirements.txt` run the keyword mode unless the `rag` extra is installed and
`HOTEL_RAG_MODE=semantic` is set.

## 12. Repository layout

```text
app.py                         Hugging Face Spaces entry point
app/gradio_app.py              Gradio UI factory
configs/booking.json           Required slots and turn policy
data/intent/                   Synthetic multilingual intent data and provenance
data/ner/                      Synthetic BIO data and provenance
models/trained/                Trained classifiers and measured development results
reports/                       Generated evaluation-output location
src/hotel_assistant/
  dialogue/state.py            Deterministic per-session booking state
  services/booking.py           Guarded booking sequence and quote
  ml/                           Training and guarded inference
  rag/policies.py              Grounded policy retrieval
  services/chat.py             Conversation controller
  services/interpret.py        Conservative slot parser
tests/{unit,ml,integration}/   Automated checks by layer
```

See [docs/roadmap.md](docs/roadmap.md) for the ordered next phases.
