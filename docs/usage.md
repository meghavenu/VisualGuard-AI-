# Usage Guide

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
Copy-Item .env.example .env
streamlit run app.py
```

Add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

## Level 1

Select **Level 1 · Screenshot Audit**, upload a screenshot, and run the audit.

## Level 2

Select **Level 2 · Before/After Regression** and provide baseline/current screenshots.

Optional ignored rectangles use:

```text
x,y,width,height
```

Use one region per line.

## Level 3

Select **Level 3 · Autonomous Website Scan** and define pages as:

```text
home|https://example.com
wiki_ai|https://www.wikipedia.org/wiki/Artificial_intelligence
wiki_ux|https://www.wikipedia.org/wiki/User_experience
```

The first run creates missing baselines. Later runs compare new captures against stored baselines.

Use **Refresh approved baselines** only after reviewing and intentionally approving the current UI.

## Optional login

The scanner supports a login URL, username, password, and CSS selectors for the username field, password field, and submit button.

Use test credentials whenever possible. Do not hard-code credentials into source files.

## Regression threshold

The Level 3 threshold is the maximum changed-area percentage tolerated before a page is marked `regression_detected`.

For example, with a threshold of `2.0`, a measured diff above 2% is treated as a regression.

## Gemini review

Gemini review is optional in Level 3 because each compared page can consume model quota.

The application has deterministic fallback behavior when the API key is missing or a model response cannot be parsed.

## Outputs

The UI exposes JSON, Markdown, TXT, and CSV representations of the generated analysis.
