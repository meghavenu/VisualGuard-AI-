# 🛡️ VisualGuard AI

**AI-powered visual regression testing and UI quality assurance for modern web applications.**

VisualGuard AI combines **computer vision, automated browser testing, pixel-level comparison, and multimodal AI** to identify visual changes, localize regressions, and generate structured QA reports.

---

## 🚀 Overview

VisualGuard AI is a three-level visual QA system designed to move from **single-screenshot design analysis** to **automated website-wide regression scanning**.

It combines deterministic image analysis with AI-based interpretation, making visual testing measurable, repeatable, and easier to review.

### What it solves

- Detects meaningful visual changes between UI states
- Identifies where regressions occur on a page
- Evaluates screenshot-based UI/UX issues
- Automates browser screenshot capture
- Handles dynamic page regions during comparison
- Generates structured, review-ready QA reports
- Supports approved baseline management and versioning

---

## ✨ Key Features

### 1. Screenshot-Based UI/UX Audit

Analyze a UI screenshot for:

- Visual hierarchy
- Contrast
- Spacing
- Alignment
- Design consistency

Findings are organized with severity, location, user impact, recommendation, evidence, and confidence.

### 2. Visual Regression Detection

Compare a **baseline** screenshot with a **current** screenshot using:

- SSIM-based similarity analysis
- OpenCV image processing
- Threshold-based change detection
- Contour-based region localization
- Changed-area measurement
- Highlighted visual diff generation

### 3. Autonomous Website Scanning

Scan multiple live web pages automatically with Playwright.

The scanner:

1. Opens configured pages
2. Captures screenshots at a consistent viewport
3. Creates or loads approved visual baselines
4. Compares current captures against baselines
5. Filters configured dynamic regions
6. Detects meaningful visual regressions
7. Produces page-level and aggregate results

### 4. AI-Assisted Visual Review

Gemini Vision provides semantic interpretation of detected changes and screenshot findings, complementing deterministic computer-vision measurements.

### 5. Structured Reporting

Generate QA results in:

- JSON
- Markdown
- TXT
- CSV

This makes the output suitable for both human review and downstream automation.

### 6. Baseline Management

Approved baselines can be refreshed while preserving previous versions, creating a traceable visual reference history.

---

## 🏗️ Architecture

```text
                         ┌─────────────────────────┐
                         │       Streamlit UI       │
                         │      VisualGuard AI      │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
                ▼                     ▼                     ▼
       ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
       │ Level 1        │   │ Level 2          │   │ Level 3          │
       │ Screenshot     │   │ Before / After   │   │ Autonomous       │
       │ Audit          │   │ Regression       │   │ Website Scan     │
       └───────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
               │                      │                      │
               ▼                      ▼                      ▼
       ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
       │ Gemini Vision  │   │ SSIM + OpenCV    │   │ Playwright       │
       │ + Image        │   │ + Contours       │   │ Screenshot       │
       │ Analysis       │   │ + Pixel Diff     │   │ Capture          │
       └───────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
               │                      │                      │
               └──────────────────────┼──────────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │ Visual Change Analysis  │
                         │ + AI Interpretation     │
                         └────────────┬────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │ Structured QA Reports   │
                         │ JSON / MD / TXT / CSV   │
                         └─────────────────────────┘
```

---

## 🔬 Three-Level QA Pipeline

| Level | Purpose | Core Technologies |
|---|---|---|
| **Level 1** | Screenshot-based UI/UX analysis | Gemini Vision, Pillow, NumPy, OpenCV |
| **Level 2** | Baseline vs. current regression testing | SSIM, OpenCV, NumPy |
| **Level 3** | Automated multi-page visual regression scanning | Playwright, OpenCV, SSIM, Gemini |

### Level 1 — Screenshot Audit

```text
UI Screenshot
      ↓
Image / Contrast Analysis
      ↓
Gemini Vision Review
      ↓
Structured Findings
      ↓
QA Report
```

### Level 2 — Visual Regression

```text
Baseline Screenshot ──┐
                      ├── Image Normalization
Current Screenshot ───┘
                              ↓
                         SSIM Analysis
                              ↓
                      Pixel / Contour Diff
                              ↓
                    Changed Region Detection
                              ↓
                    Visual Diff + QA Report
```

### Level 3 — Autonomous Scan

```text
Configured URLs
      ↓
Playwright Browser Automation
      ↓
Full-Page Screenshots
      ↓
Baseline Management
      ↓
Visual Regression Engine
      ↓
Dynamic Region Filtering
      ↓
Regression Decision
      ↓
AI-Assisted Review
      ↓
Aggregate Scan Report
```

---

## 🧰 Tech Stack

### Frontend & Application

- **Python**
- **Streamlit**

### Computer Vision & Image Analysis

- **OpenCV**
- **NumPy**
- **Pillow**
- **SSIM**

### AI

- **Google Gemini Vision**

### Browser Automation

- **Playwright**
- **Chromium**

### Reporting & Data

- **JSON**
- **Markdown**
- **CSV**
- **TXT**

### Development & Automation

- **GitHub Actions**

---

## 📁 Project Structure

```text
VisualGuard-AI/
│
├── app.py
├── capture_page.py
├── requirements.txt
│
├── .env.example
├── .gitignore
├── LICENSE
│
├── README.md
├── PROJECT_STRUCTURE.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
│
├── docs/
│   ├── architecture.md
│   ├── usage.md
│   └── assets/
│
├── legacy/
│
├── baselines/
├── screenshots/
├── outputs/
├── reports/
├── uploads/
└── logs/
```

Runtime directories are maintained for application output while generated artifacts remain excluded from source control.

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd VisualGuard-AI
```

### 2. Create the environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure the application

Configure the environment values required by the application.

### 5. Launch

```bash
streamlit run app.py
```

---

## 📊 Output

VisualGuard produces both visual evidence and structured results, including:

- Similarity scores
- Changed-area percentages
- Detected regions
- Regression status
- Confidence levels
- AI-generated observations
- Recommendations
- Machine-readable result files

---

## 🎯 Why VisualGuard AI?

Traditional visual QA often depends on manually comparing screenshots after every UI change.

VisualGuard AI introduces an automated pipeline that combines:

**Browser Automation → Screenshot Capture → Computer Vision → Regression Detection → AI Interpretation → Structured Reporting**

This creates a repeatable workflow for identifying visual regressions before they reach users.

---

## 🔮 Future Enhancements

- CI/CD visual regression gates
- Multi-browser comparison
- Component-level regression tracking
- Historical regression dashboards
- Automated notification workflows
- Expanded accessibility analysis
- Parallelized website scanning

---

## 📄 License

This project is licensed under the **MIT License**.
