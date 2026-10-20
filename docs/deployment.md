# Hugging Face Space deployment

The public academic demo is packaged separately from the private GitHub repository. The Space
bundle contains only the Gradio entry point, application modules, authored hotel policies,
configuration, and trained classifier artifacts. It excludes the local Qwen model download,
evaluation reports, local SQLite database, and development files.

The Space runs the trained intent model and semantic FAISS policy retrieval. Qwen writes
short replies from verified policy and conversation text on the Space CPU. Booking
quotes, reservation creation, and confirmation remain deterministic. If generation fails or
violates the reply checks, the verified text is shown instead. Reservations are fictional and use a temporary
SQLite database that can disappear when the Space restarts. No payment is processed.

The account uses a free ZeroGPU Space because Hugging Face currently requires a PRO plan to
create Gradio Spaces on CPU Basic. ZeroGPU specifies Python 3.12.12 and adds Gradio extras that
require Pydantic 2.12.5, so the Space has its own dependency list. The application's source
modules are placed on the Python path directly by its entry point. Conversation state stays in
the app process. Policy retrieval stays on CPU; Qwen weights are preloaded from the Hub at
build time. CPU generation is slower, but it avoids the unavailable ZeroGPU workers seen in
live tests.

To rebuild the exact public bundle:

```bash
python scripts/prepare_hf_space.py
hf upload itsmarlo/bellhob-hotel-chatbot /tmp/hotel-chatbot-hf-space . \
  --type space --commit-message 'Update academic demo'
hf spaces wait itsmarlo/bellhob-hotel-chatbot
```

The Space files are generated from `deployment/huggingface/` and selected source paths by
`scripts/prepare_hf_space.py`. Review that directory before each public upload. The Space
rebuilds automatically after an upload.

On 20 October 2026, the public URL returned HTTP 200. Live Gradio API checks returned a cited
check-in policy, retrieved the same source for a paraphrase without an exact keyword, and
created a synthetic reservation only after a second-turn confirmation.
