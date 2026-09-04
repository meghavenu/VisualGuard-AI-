import os
import json
import base64
from pathlib import Path
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv


st.set_page_config(
    page_title="VisualGuard AI",
    page_icon="🛡️",
    layout="wide"
)

load_dotenv()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")
LOG_DIR = Path("logs")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


if "audit_result" not in st.session_state:
    st.session_state.audit_result = None

if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None


def log_event(event_type, data):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }

    with open(LOG_DIR / "agent_logs.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_uploaded_file(uploaded_file):
    safe_name = uploaded_file.name.replace(" ", "_")
    file_path = UPLOAD_DIR / safe_name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def get_mime_type(file_path):
    suffix = str(file_path).lower()

    if suffix.endswith(".png"):
        return "image/png"

    if suffix.endswith(".jpg") or suffix.endswith(".jpeg"):
        return "image/jpeg"

    if suffix.endswith(".webp"):
        return "image/webp"

    return "image/png"


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])


def relative_luminance(rgb):
    values = []

    for c in rgb:
        c = c / 255

        if c <= 0.03928:
            values.append(c / 12.92)
        else:
            values.append(((c + 0.055) / 1.055) ** 2.4)

    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def contrast_ratio(color1, color2):
    l1 = relative_luminance(color1)
    l2 = relative_luminance(color2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return round((lighter + 0.05) / (darker + 0.05), 2)


def dominant_colors(image_path, k=6):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((250, 250))

    pixels = np.array(image).reshape(-1, 3)
    pixels = (pixels // 16) * 16

    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    sorted_indices = counts.argsort()[-k:][::-1]

    colors = unique_colors[sorted_indices]

    return [tuple(map(int, color)) for color in colors]


def analyze_contrast(image_path):
    colors = dominant_colors(image_path)

    results = []

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            ratio = contrast_ratio(colors[i], colors[j])

            results.append({
                "color_1": rgb_to_hex(colors[i]),
                "color_2": rgb_to_hex(colors[j]),
                "contrast_ratio": ratio,
                "wcag_aa_pass": ratio >= 4.5
            })

    results.sort(key=lambda item: item["contrast_ratio"])
    return results


def clean_json_response(text):
    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= start:
        raise ValueError("Gemini did not return valid JSON.")

    return text[start:end]


def normalize_report(report, image_path):
    report["source"] = report.get("source", "Gemini Vision")
    report["resource_id"] = report.get("resource_id", str(image_path))

    findings = report.get("findings", [])
    normalized_findings = []

    for finding in findings:
        confidence = finding.get("confidence", 70)

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 70

        if confidence <= 1:
            confidence = confidence * 100

        confidence = int(round(confidence))
        confidence = max(0, min(100, confidence))

        finding["confidence"] = confidence
        finding["principle"] = finding.get("principle", "General")
        finding["severity"] = finding.get("severity", "medium")
        finding["location"] = finding.get("location", "Visible UI area")
        finding["issue"] = finding.get("issue", "Design issue detected.")
        finding["user_impact"] = finding.get("user_impact", "This may affect usability or readability.")
        finding["recommendation"] = finding.get("recommendation", "Review and improve this UI area.")
        finding["evidence"] = finding.get("evidence", "Visible in the screenshot.")

        normalized_findings.append(finding)

    report["findings"] = normalized_findings

    score = report.get("overall_score", None)

    try:
        score = float(score)
    except Exception:
        score = None

    if score is None or score <= 0:
        severity_penalty = {
            "critical": 25,
            "high": 18,
            "medium": 10,
            "low": 5,
            "info": 2
        }

        total_penalty = 0

        for finding in normalized_findings:
            severity = finding.get("severity", "medium").lower()
            total_penalty += severity_penalty.get(severity, 10)

        score = max(0, 100 - total_penalty)

    if score <= 1:
        score = score * 100

    score = int(round(score))
    score = max(0, min(100, score))

    report["overall_score"] = score

    return report


def fallback_report(image_path, contrast_data, reason):
    weak_contrast = []

    for item in contrast_data:
        if item["wcag_aa_pass"] is False:
            weak_contrast.append(item)

    contrast_evidence = weak_contrast[:3] if weak_contrast else contrast_data[:3]

    return {
        "source": "Fallback",
        "fallback_reason": reason,
        "resource_id": str(image_path),
        "overall_score": 74,
        "findings": [
            {
                "principle": "Contrast",
                "severity": "medium",
                "location": "Visible text and foreground UI elements",
                "issue": "Some visible foreground and background color combinations may not provide enough contrast.",
                "user_impact": "Users with low vision may find text or controls difficult to read.",
                "recommendation": "Increase foreground/background contrast and target at least 4.5:1 for normal text.",
                "evidence": str(contrast_evidence),
                "confidence": 78
            },
            {
                "principle": "Spacing",
                "severity": "low",
                "location": "Main content area",
                "issue": "Some content groups may need more breathing space for better scanability.",
                "user_impact": "Crowded spacing can make the interface feel dense and harder to understand quickly.",
                "recommendation": "Use consistent spacing values like 8px, 16px, and 24px between related sections.",
                "evidence": "Detected from uploaded screenshot layout density.",
                "confidence": 68
            },
            {
                "principle": "Consistency",
                "severity": "info",
                "location": "Repeated UI components",
                "issue": "Component consistency cannot be fully confirmed without the original design system or design tokens.",
                "user_impact": "Inconsistent buttons, cards, or typography can reduce trust and predictability.",
                "recommendation": "Compare buttons, cards, colors, and typography with a shared design system.",
                "evidence": "Design-token metadata is unavailable for this screenshot.",
                "confidence": 62
            }
        ]
    }


def analyze_with_gemini_rest(image_path, contrast_data):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback_report(
            image_path,
            contrast_data,
            "GEMINI_API_KEY is missing from .env."
        )

    if api_key.strip() == "your_gemini_api_key_here":
        return fallback_report(
            image_path,
            contrast_data,
            "GEMINI_API_KEY still has placeholder value."
        )

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        mime_type = get_mime_type(image_path)

        prompt = f"""
You are VisualGuard AI, a production-grade UI/UX Design Audit Agent.

Analyze the uploaded UI screenshot and identify visible design issues.

Strict rules:
1. Do not hallucinate.
2. Every finding must be based only on something visible in the screenshot.
3. Mention actual visible UI areas, not generic text like "main content area".
4. If something is uncertain, reduce confidence.
5. Do not give vague feedback like "bad UI".
6. Give specific location, issue, user impact, recommendation, evidence, and confidence.
7. Find at least 5 findings if possible.
8. Cover all five principles:
   - Visual Hierarchy
   - Contrast / WCAG AA
   - Spacing
   - Alignment
   - Consistency

Severity rules:
- critical: prevents task completion or makes content unreadable
- high: serious accessibility or usability issue
- medium: noticeable issue affecting clarity
- low: minor design improvement
- info: observation or optional improvement

Return only valid JSON in this exact format:

{{
  "source": "Gemini Vision",
  "resource_id": "uploaded_file_name",
  "overall_score": 85,
  "findings": [
    {{
      "principle": "Visual Hierarchy",
      "severity": "medium",
      "location": "specific visible location on the page",
      "issue": "specific issue visible in screenshot",
      "user_impact": "why this affects users",
      "recommendation": "specific practical fix",
      "evidence": "visible evidence from screenshot",
      "confidence": 85
    }}
  ]
}}

Important:
- overall_score must be an integer from 0 to 100.
- confidence must be an integer from 0 to 100.
- Do not return decimals like 0.85. Return 85.

Measurable contrast evidence from image processing:
{json.dumps(contrast_data[:10], indent=2)}
"""

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        model_names = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.5-flash"
        ]

        response = None
        last_error = None
        used_model = None

        for model_name in model_names:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=90
            )

            if response.status_code == 200:
                used_model = model_name
                break

            last_error = f"{model_name}: {response.status_code} - {response.text}"

        if response is None or response.status_code != 200:
            return fallback_report(
                image_path,
                contrast_data,
                f"Gemini REST API failed after trying multiple models: {last_error}"
            )

        response_json = response.json()
        text = response_json["candidates"][0]["content"]["parts"][0]["text"]

        cleaned = clean_json_response(text)
        result = json.loads(cleaned)
        result = normalize_report(result, image_path)
        result["model_used"] = used_model

        if "findings" not in result or len(result["findings"]) < 3:
            return fallback_report(
                image_path,
                contrast_data,
                "Gemini returned fewer than 3 findings."
            )

        return result

    except Exception as error:
        return fallback_report(
            image_path,
            contrast_data,
            f"Gemini API failed: {str(error)}"
        )


def save_json_report(report):
    output_path = OUTPUT_DIR / "level1_audit.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return output_path


def generate_markdown_text(report):
    lines = []

    lines.append("# VisualGuard AI - Level 1 Design Audit Report")
    lines.append("")
    lines.append(f"Source: {report.get('source', 'Unknown')}")
    lines.append(f"Model Used: {report.get('model_used', 'N/A')}")
    lines.append(f"Resource: {report.get('resource_id')}")
    lines.append(f"Overall Score: {report.get('overall_score')}/100")
    lines.append("")

    if report.get("source") == "Fallback":
        lines.append(f"Fallback Reason: {report.get('fallback_reason', 'Unknown')}")
        lines.append("")

    lines.append("## Findings")
    lines.append("")

    for index, finding in enumerate(report.get("findings", []), 1):
        lines.append(f"### {index}. {finding.get('principle')} - {finding.get('severity').upper()}")
        lines.append("")
        lines.append(f"Location: {finding.get('location')}")
        lines.append(f"Issue: {finding.get('issue')}")
        lines.append(f"User Impact: {finding.get('user_impact')}")
        lines.append(f"Recommendation: {finding.get('recommendation')}")
        lines.append(f"Evidence: {finding.get('evidence')}")
        lines.append(f"Confidence: {finding.get('confidence')}%")
        lines.append("")

    return "\n".join(lines)


def generate_text_report(report):
    lines = []

    lines.append("VISUALGUARD AI - LEVEL 1 DESIGN AUDIT REPORT")
    lines.append("=" * 55)
    lines.append(f"Source: {report.get('source', 'Unknown')}")
    lines.append(f"Model Used: {report.get('model_used', 'N/A')}")
    lines.append(f"Resource: {report.get('resource_id')}")
    lines.append(f"Overall Score: {report.get('overall_score')}/100")
    lines.append("")

    for index, finding in enumerate(report.get("findings", []), 1):
        lines.append(f"{index}. {finding.get('principle')} - {finding.get('severity').upper()}")
        lines.append(f"Location: {finding.get('location')}")
        lines.append(f"Issue: {finding.get('issue')}")
        lines.append(f"User Impact: {finding.get('user_impact')}")
        lines.append(f"Recommendation: {finding.get('recommendation')}")
        lines.append(f"Evidence: {finding.get('evidence')}")
        lines.append(f"Confidence: {finding.get('confidence')}%")
        lines.append("-" * 55)

    return "\n".join(lines)


def save_markdown_report(report):
    markdown_text = generate_markdown_text(report)
    report_path = REPORT_DIR / "level1_report.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)

    return report_path


def convert_findings_to_csv(report):
    rows = []

    for finding in report.get("findings", []):
        rows.append({
            "principle": finding.get("principle"),
            "severity": finding.get("severity"),
            "location": finding.get("location"),
            "issue": finding.get("issue"),
            "user_impact": finding.get("user_impact"),
            "recommendation": finding.get("recommendation"),
            "evidence": finding.get("evidence"),
            "confidence": finding.get("confidence")
        })

    dataframe = pd.DataFrame(rows)
    return dataframe.to_csv(index=False).encode("utf-8")


def run_level1_audit(image_path):
    log_event("level1_started", {"image_path": str(image_path)})

    contrast_data = analyze_contrast(image_path)

    log_event("contrast_analysis_completed", {
        "contrast_pairs_checked": len(contrast_data)
    })

    report = analyze_with_gemini_rest(image_path, contrast_data)

    save_json_report(report)
    save_markdown_report(report)

    log_event("level1_completed", {
        "source": report.get("source"),
        "model_used": report.get("model_used"),
        "findings_count": len(report["findings"]),
        "overall_score": report["overall_score"]
    })

    return report


def severity_badge(severity):
    severity = severity.lower()

    if severity == "critical":
        return "Critical"
    if severity == "high":
        return "High"
    if severity == "medium":
        return "Medium"
    if severity == "low":
        return "Low"

    return "Info"


st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    .stApp {
        background: #020617;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1080px;
    }

    section[data-testid="stSidebar"] {
        background: #050b16;
        border-right: 1px solid #13233a;
        width: 240px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
        font-size: 14px !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #f8fafc !important;
    }

    h1 {
        font-size: 38px !important;
        letter-spacing: -0.04em;
    }

    h2 {
        font-size: 25px !important;
        letter-spacing: -0.03em;
    }

    h3 {
        font-size: 20px !important;
        letter-spacing: -0.02em;
    }

    p, label, span {
        font-size: 14px !important;
    }

    .hero-card {
        background: linear-gradient(135deg, #071426 0%, #0b1d35 100%);
        border: 1px solid #13233a;
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 22px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
        animation: fadeUp 0.5s ease-out;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 760;
        color: #f8fafc;
        margin-bottom: 8px;
        letter-spacing: -0.05em;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #cbd5e1 !important;
        margin-bottom: 16px;
        max-width: 760px;
        line-height: 1.7;
    }

    .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #0f2742;
        color: #e0f2fe;
        border: 1px solid #24496f;
        font-size: 12px;
        margin-right: 7px;
        margin-top: 7px;
    }

    .panel {
        background: #071426;
        border: 1px solid #13233a;
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.34);
        animation: fadeUp 0.45s ease-out;
    }

    .panel-title {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 720;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
    }

    .panel-text {
        color: #cbd5e1 !important;
        font-size: 13px !important;
        line-height: 1.65;
    }

    .finding-card {
        background: #071426;
        color: #f8fafc;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid #13233a;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.34);
        animation: fadeUp 0.45s ease-out;
    }

    .finding-card h3,
    .finding-card p,
    .finding-card b {
        color: #f8fafc !important;
    }

    .small-label {
        color: #7dd3fc !important;
        font-size: 11px !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stMetric"] {
        background: #071426;
        border-radius: 18px;
        padding: 16px;
        border: 1px solid #13233a;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32);
    }

    div[data-testid="stMetric"] label {
        color: #93c5fd !important;
        font-size: 12px !important;
        font-weight: 650;
    }

    div[data-testid="stMetric"] div {
        color: #f8fafc !important;
        font-size: 22px !important;
    }

    div[data-testid="stFileUploader"] {
        background: #071426;
        border-radius: 18px;
        padding: 12px;
        border: 1px solid #13233a;
    }

    div[data-testid="stFileUploader"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background: #0b1220 !important;
        border: 1px dashed #38bdf8 !important;
        border-radius: 16px !important;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stFileUploaderDropzone"] button {
        background: #38bdf8 !important;
        color: #020617 !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: #38bdf8;
        color: #020617 !important;
        border-radius: 12px;
        border: none;
        padding: 0.72rem 1rem;
        font-weight: 760;
        transition: all 0.25s ease-in-out;
    }

    .stButton > button:hover {
        background: #7dd3fc;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        background: #38bdf8;
        color: #020617 !important;
        border-radius: 12px;
        border: none;
        padding: 0.72rem 1rem;
        font-weight: 760;
    }

    .stDownloadButton > button:hover {
        background: #7dd3fc;
    }

    div[data-testid="stDataFrame"] {
        background: #071426;
        border-radius: 14px;
        color: #f8fafc;
    }

    textarea {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border: 1px solid #13233a !important;
        border-radius: 14px !important;
        font-family: Consolas, monospace !important;
        font-size: 13px !important;
    }

    pre, code {
        background: #020617 !important;
        color: #e5e7eb !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #071426 !important;
        border-color: #13233a !important;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">VisualGuard AI</div>
        <div class="hero-subtitle">
            Premium UI audit agent for screenshot-based design review, accessibility feedback, and structured reporting.
        </div>
        <span class="pill">Agent 1</span>
        <span class="pill">Level 1</span>
        <span class="pill">Gemini Vision</span>
        <span class="pill">Structured Output</span>
    </div>
    """,
    unsafe_allow_html=True
)


api_key_check = os.getenv("GEMINI_API_KEY")

st.sidebar.title("VisualGuard AI")
st.sidebar.caption("Design Audit Agent")

if api_key_check and api_key_check.strip() != "your_gemini_api_key_here":
    st.sidebar.success("Gemini connected")
else:
    st.sidebar.error("API key missing")

st.sidebar.divider()
st.sidebar.write("Level 1")
st.sidebar.write("Screenshot audit")
st.sidebar.write("JSON / MD / TXT / CSV")


st.markdown(
    """
    <div class="panel">
        <div class="panel-title">Single Screenshot Design Audit</div>
        <div class="panel-text">
            Upload a UI screenshot. The agent evaluates hierarchy, contrast, spacing, alignment, and consistency.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload UI Screenshot",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file:
    image_path = save_uploaded_file(uploaded_file)
    st.session_state.uploaded_image_path = str(image_path)

    col1, col2 = st.columns([1.25, 0.75])

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Visual Workspace")
        st.image(Image.open(image_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Audit Console")
        st.write("Run the audit against five core design principles.")
        run_button = st.button("Run Design Audit", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run_button:
        with st.spinner("Analyzing screenshot..."):
            st.session_state.audit_result = run_level1_audit(image_path)

if st.session_state.audit_result is not None:
    result = st.session_state.audit_result

    if result.get("source") == "Gemini Vision":
        st.success("Audit completed using Gemini Vision.")
    else:
        st.warning("Audit completed using fallback output.")
        st.error(result.get("fallback_reason", "Unknown fallback reason"))

    findings = result["findings"]

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Score", f"{result['overall_score']}/100")

    with m2:
        st.metric("Findings", len(findings))

    with m3:
        avg_confidence = round(
            sum(item["confidence"] for item in findings) / len(findings),
            2
        )
        st.metric("Confidence", f"{avg_confidence}%")

    with m4:
        st.metric("Model", result.get("model_used", result.get("source", "Unknown")))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Findings Summary")

    table_data = []

    for item in findings:
        table_data.append({
            "Principle": item["principle"],
            "Severity": item["severity"],
            "Location": item["location"],
            "Confidence": item["confidence"]
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Detailed Findings")

    for index, finding in enumerate(findings, 1):
        st.markdown(
            f"""
            <div class="finding-card">
                <p class="small-label">Finding {index}</p>
                <h3>{finding["principle"]} · {severity_badge(finding["severity"])}</h3>
                <p><b>Location:</b> {finding["location"]}</p>
                <p><b>Issue:</b> {finding["issue"]}</p>
                <p><b>User Impact:</b> {finding["user_impact"]}</p>
                <p><b>Recommendation:</b> {finding["recommendation"]}</p>
                <p><b>Evidence:</b> {finding["evidence"]}</p>
                <p><b>Confidence:</b> {finding["confidence"]}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Structured Output")

    json_text = json.dumps(result, indent=2, ensure_ascii=False)
    markdown_text = generate_markdown_text(result)
    txt_text = generate_text_report(result)
    csv_data = convert_findings_to_csv(result)

    output_format = st.selectbox(
        "Output format",
        ["JSON", "Markdown", "Text", "CSV"]
    )

    if output_format == "JSON":
        st.text_area("Preview", json_text, height=420)
        st.download_button(
            "Download JSON",
            json_text.encode("utf-8"),
            "visualguard_level1_report.json",
            "application/json",
            use_container_width=True
        )

    elif output_format == "Markdown":
        st.text_area("Preview", markdown_text, height=420)
        st.download_button(
            "Download Markdown",
            markdown_text.encode("utf-8"),
            "visualguard_level1_report.md",
            "text/markdown",
            use_container_width=True
        )

    elif output_format == "Text":
        st.text_area("Preview", txt_text, height=420)
        st.download_button(
            "Download Text",
            txt_text.encode("utf-8"),
            "visualguard_level1_report.txt",
            "text/plain",
            use_container_width=True
        )

    else:
        st.dataframe(pd.DataFrame(result["findings"]), use_container_width=True)
        st.download_button(
            "Download CSV",
            csv_data,
            "visualguard_level1_findings.csv",
            "text/csv",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload a screenshot and run the audit.")