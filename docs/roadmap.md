# Bellhob implementation roadmap

The Phase 1 concept for the fictional Beit Zeitouna hotel is the product plan. Its order is:
guarded state and booking tools, policy retrieval, the multilingual LLM layer, the Gradio
interface, then a single-prompt baseline and comparison.

## Phase 2: guarded booking path (in progress)

- Done: deterministic state holds name, dates, guest count, optional room preference and request.
- Done: search rooms, calculate a quote, require an explicit confirmation, then create a booking.
- Done: recheck occupancy and quoted nightly price in one SQLite write transaction.
- Done: tests for invalid data, stale quotes, changed prices, occupied rooms and confirmation.
- Next: expose this workflow through a conversation controller. Pass the raw user confirmation
  message to the guard; never let an LLM invent confirmation on the user's behalf.
- Next: define a currency, cancellation policy, and any price/tax rules before showing a
  customer-facing total. The current integer amounts are synthetic demo cents.

## Phase 3: hotel-policy retrieval

- Author fictional English, French and German policy/FAQ documents with source metadata.
- Chunk and embed them using multilingual Sentence-Transformers and a local FAISS index.
- Evaluate paraphrased questions separately from source chunks; decline unsupported answers.

## Phase 4: LLM interpretation and conversation controller

- Use Qwen2.5-1.5B-Instruct to propose intent, slots and tool calls in constrained JSON.
- Cross-check extracted dates with a deterministic parser and clarify disagreements.
- Route each turn to exactly one of booking, policy knowledge, or clarification.
- Keep state and tool order authoritative; the model must not write reservations directly.
- Measure CPU memory and latency before choosing an inference deployment method.

## Phase 5: Gradio and comparison

- Connect the controller to the existing per-session Gradio shell and display grounded sources.
- Build a single-prompt baseline that runs on the same scripted dialogues but cannot alter
  the guarded booking store.
- Evaluate 30 dialogues, 10 per language, against the concept's completion, invalid-input,
  pre-confirmation, retrieval and unsafe-booking targets. Generate metrics from test runs.

The earlier intent and BIO starter datasets remain available as experiments; they do not replace
the multilingual LLM design selected in the Phase 1 concept.
