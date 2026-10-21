# Hugging Face Space deployment

The public academic demo is packaged separately from the private GitHub repository. The Space
bundle contains only the Gradio entry point, application modules, authored hotel policies,
configuration, and trained classifier artifacts. It excludes the local Qwen model download,
evaluation reports, local SQLite database, and development files.

The Space runs the trained intent model and semantic FAISS policy retrieval. An optional hosted
router can rephrase verified policy answers through Hugging Face, OpenRouter, or OpenAI. It tries
configured models in order and falls through on provider errors, timeouts, empty output, or an
invalid response shape. Booking prompts, quotes, reservation creation, and confirmation remain
deterministic. If generation is unavailable or violates the reply checks, the verified text is
shown instead. Reservations are fictional and use a temporary SQLite database that can disappear
when the Space restarts. No payment is processed.

The account uses a free ZeroGPU Space because Hugging Face currently requires a PRO plan to
create Gradio Spaces on CPU Basic. ZeroGPU specifies Python 3.12.12 and adds Gradio extras that
require Pydantic 2.12.5, so the Space has its own dependency list. The application's source
modules are placed on the Python path directly by its entry point. Conversation state stays in
the app process. Policy query embeddings try ZeroGPU once and fall back to CPU if it is
unavailable. No generative model weights are loaded into the Space process.

Configure one or more Space secrets to enable hosted generation:

- `HF_TOKEN`: dedicated fine-grained Hugging Face token with Inference Providers permission
- `OPENROUTER_API_KEY`: OpenRouter credential
- `OPENAI_API_KEY`: OpenAI API credential

Optional Space variables control routing: `HOTEL_LLM_ORDER`, `HOTEL_HF_MODELS`,
`HOTEL_OPENROUTER_MODELS`, `HOTEL_OPENAI_MODELS`, and `HOTEL_LLM_TIMEOUT_SECONDS`. Secrets are
read at runtime and are never copied into the public Space files. Without a valid secret and
provider credit, the application remains usable and returns verified template answers.

To rebuild the exact public bundle:

```bash
python scripts/prepare_hf_space.py
hf upload itsmarlo/bellhob-hotel-chatbot /tmp/hotel-chatbot-hf-space . \
  --type space --delete data/synthetic/hotel.db \
  --commit-message 'Update academic demo'
hf spaces wait itsmarlo/bellhob-hotel-chatbot
```

The Space files are generated from `deployment/huggingface/` and selected source paths by
`scripts/prepare_hf_space.py`. Review that directory before each public upload. The Space
rebuilds automatically after an upload.

On 20 October 2026, the public URL returned HTTP 200. Live Gradio API checks returned a cited
check-in policy, retrieved the same source for a paraphrase without an exact keyword, and
created a synthetic reservation only after a second-turn confirmation.

On 21 October 2026, Space revision `99a859bf3e340cc6960aa7a88ff40fc45d6839a6` reached
`RUNNING`. The public repository no longer includes the seed database, and a live check-in
question returned the verified 15:00 and 11:00 times with the `[check-in]` citation.
