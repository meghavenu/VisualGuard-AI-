# VisualGuard AI — Technical Architecture

## 1. System overview

VisualGuard AI is implemented as a Streamlit application with a primary entry point (`app.py`) and a dedicated Playwright capture helper (`capture_page.py`).

The architecture is intentionally lightweight: deterministic visual analysis runs locally, while Gemini is an optional semantic-analysis layer.

## 2. Processing pipeline

### Level 1

```text
Screenshot
   │
   ├── Pillow / NumPy
   │      └── dominant-color extraction
   │              └── contrast-ratio analysis
   │
   └── Gemini Vision (optional)
          └── structured UI/UX findings
                    │
                    ▼
             normalization + fallback
                    │
                    ▼
          JSON / Markdown / TXT / CSV
```

### Level 2

```text
Baseline ─────┐
              ├── dimension normalization
Current ──────┘
              │
              ├── ignored-region masking
              ├── grayscale conversion
              ├── SSIM similarity
              ├── thresholding
              ├── contour detection
              └── changed-region measurement
                         │
                         ├── highlighted diff image
                         └── optional Gemini interpretation
```

### Level 3

```text
Page definitions
      │
      ▼
Playwright capture
      │
      ├── optional login
      └── fixed 1440×900 viewport
      │
      ▼
Current screenshot
      │
      ├── no baseline → create baseline
      │
      └── baseline exists
              │
              ▼
        Level 2 diff engine
              │
              ├── threshold decision
              └── optional Gemini review
                      │
                      ▼
                page-level result
                      │
                      ▼
              aggregate scan report
```

## 3. Key design decisions

### Deterministic + AI hybrid

Pixel-level comparison is deterministic and reproducible. Gemini is used for semantic interpretation rather than replacing the measurable diff engine.

### Graceful degradation

When Gemini is unavailable, the application falls back to deterministic evidence-based findings instead of failing the entire workflow.

### Baseline versioning

Refreshing an existing baseline copies the previous baseline into `baselines/versions/` with a timestamp before replacing it.

### Dynamic-content handling

Users can specify rectangular regions that should be ignored during visual comparison. Small detected contours are also filtered to reduce noise.

### Structured outputs

Results are written as machine-readable JSON and human-readable Markdown, with TXT/CSV representations exposed by the UI.

## 4. Runtime storage

| Directory | Purpose |
|---|---|
| `baselines/` | Approved screenshots used as visual references |
| `baselines/versions/` | Timestamped previous baselines |
| `screenshots/current/` | Latest Playwright captures |
| `screenshots/diff/` | Highlighted visual diffs |
| `outputs/` | JSON result artifacts |
| `reports/` | Markdown report artifacts |
| `uploads/` | User-uploaded screenshots |
| `logs/` | Runtime logs |

Runtime artifacts are intentionally ignored by Git in the public repository.
