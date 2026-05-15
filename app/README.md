# pgx-digest — Streamlit demo

A 30-second click-through of the pgx-digest thesis: a deterministic
PharmCAT parse feeds a fenced LLM Drafter, whose output is checked
field-by-field against the source Bundle by a typed Verifier. The
"Try it" tab attaches a green-check badge to each verified field so the
Verifier's role is visually concrete; the "Ablations" tab embeds the
markdown tables produced by `examples/run_ablations.py`.

## Run locally

From the repo root:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Paste an Anthropic or Gemini API key in the sidebar (or set
`ANTHROPIC_API_KEY` / `GEMINI_API_KEY` in your env), pick a fixture,
and click **Generate Report**. One API call per click — no work happens
on page load.

## Deploy to Streamlit Community Cloud

Connect the repo at [share.streamlit.io](https://share.streamlit.io),
set the **Main file path** to `app/streamlit_app.py` and the
**Requirements file** to `app/requirements.txt`. The app reads keys
only from in-memory session state — do **not** put keys in
`secrets.toml` for a public deployment unless you accept the cost
exposure.
