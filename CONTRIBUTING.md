# Contributing

Thanks for contributing to VisualGuard AI.

## Guidelines

- Keep visual-analysis logic deterministic where possible.
- Treat Gemini as an optional semantic layer.
- Do not commit API keys, credentials, private screenshots, or runtime logs.
- Keep generated artifacts out of source control.
- Preserve existing JSON output fields when extending analysis where practical.
- Document new configuration options.

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
streamlit run app.py
```

## Pull requests

Please describe what changed, why it changed, how it was tested, and whether the output/report schema changed.
