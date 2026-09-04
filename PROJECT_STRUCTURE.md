# Project Structure

```text
VisualGuard-AI/
├── app.py                         # Main Streamlit application
├── capture_page.py                # Playwright screenshot helper
├── requirements.txt               # Python dependencies
├── .env.example                   # Public environment template
├── .gitignore                     # Repository hygiene
├── README.md                      # GitHub landing page
├── docs/
│   ├── architecture.md            # Technical architecture
│   ├── usage.md                   # Setup and usage guide
│   └── assets/                    # Small demo visuals
├── baselines/                     # Runtime baseline store
├── outputs/                       # Runtime JSON outputs
├── reports/                       # Runtime reports
├── screenshots/                   # Runtime screenshots/diffs
├── uploads/                       # Runtime uploads
├── logs/                          # Runtime logs
├── legacy/                        # Historical backup implementations
└── .github/
    ├── workflows/                 # CI
    ├── ISSUE_TEMPLATE/            # Issue templates
    └── PULL_REQUEST_TEMPLATE.md   # PR checklist
```
