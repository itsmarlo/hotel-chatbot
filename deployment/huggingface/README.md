---
title: Bellhob Hotel Assistant
emoji: 🛎️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: 3.12.12
app_file: app.py
preload_from_hub:
  - Qwen/Qwen2.5-1.5B-Instruct model.safetensors,config.json,generation_config.json,tokenizer.json,tokenizer_config.json,vocab.json 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
---

# Bellhob Hotel Assistant

An academic demonstration for a fictional hotel. It answers questions from authored policies
and creates synthetic reservations only after the guest confirms a quote. No payment is taken.

The running Space uses multilingual semantic retrieval and the trained intent model. Qwen
writes short replies from verified policy or conversation text. Booking quotes, confirmation,
and reservation creation remain controlled by the booking workflow. The model runs on the
Space CPU while ZeroGPU workers are unavailable, so first responses may take longer.
