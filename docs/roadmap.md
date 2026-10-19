# Bellhob implementation roadmap

The Phase 1 concept for the fictional Beit Zeitouna hotel is the product plan. Its order is:
guarded state and booking tools, policy retrieval, the multilingual LLM layer, the Gradio
interface, then a single-prompt baseline and comparison.

## Phase 2: guarded booking path (prototype complete)

- Done: deterministic state holds name, dates, guest count, optional room preference and request.
- Done: search rooms, calculate a quote, require an explicit confirmation, then create a booking.
- Done: recheck occupancy and quoted nightly price in one SQLite write transaction.
- Done: tests for invalid data, stale quotes, changed prices, occupied rooms and confirmation.
- Done: expose the workflow through the conversation controller, passing the raw user
  confirmation message to the guard.
- Next: define a currency, cancellation policy, and any price/tax rules before showing a
  customer-facing total. The current integer amounts are synthetic demo cents.

## Phase 3: hotel-policy retrieval (foundation implemented)

- Done: authored English, French and German policy answers with source metadata.
- Done: multilingual Sentence-Transformers and local FAISS option, plus a lightweight fallback.
- Done: a small authored retrieval check has 17/20 semantic Hit@1 and rejects 2/2 unsupported
  questions; this does not establish real-world accuracy.
- Next: expand the held-out set and calibrate the abstention threshold using realistic questions.

## Phase 4: LLM interpretation and conversation controller

- Done: reproducible multilingual intent baseline and BIO token classifier with fixed splits.
- Done: the intent model runs in semantic mode; token predictions are advisory only.
- Use Qwen2.5-1.5B-Instruct to propose intent, slots and tool calls in constrained JSON.
- Cross-check extracted dates with a deterministic parser and clarify disagreements.
- Done: deterministic router selects booking, policy knowledge, or clarification.
- Next: replace conservative parsing with constrained multilingual LLM proposals and date checks.
- Keep state and tool order authoritative; the model must not write reservations directly.
- Measure CPU memory and latency before choosing an inference deployment method.

## Phase 5: Gradio and comparison

- Done: connect the controller to Gradio and display grounded sources.
- Build a single-prompt baseline that runs on the same scripted dialogues but cannot alter
  the guarded booking store.
- Evaluate 30 dialogues, 10 per language, against the concept's completion, invalid-input,
  pre-confirmation, retrieval and unsafe-booking targets. Generate metrics from test runs.

The earlier intent and BIO starter datasets remain available as experiments; they do not replace
the multilingual LLM design selected in the Phase 1 concept.
