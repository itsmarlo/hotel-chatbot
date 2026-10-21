---
title: Bellhob Hotel Assistant
emoji: 🛎️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: 3.12.12
app_file: app.py
---

# Bellhob Hotel Assistant

An academic demonstration for a fictional hotel. It answers questions from authored policies
and creates synthetic reservations only after the guest confirms a quote. No payment is taken.

The running Space uses multilingual semantic retrieval and the trained intent model. A hosted
model router can rewrite verified policy text using Hugging Face, OpenRouter, or OpenAI. It
switches to the next configured model if one fails. Booking prompts, quotes, confirmation, and
reservation creation always remain controlled by the booking workflow. Without a provider
secret, the application displays its verified templates directly.
