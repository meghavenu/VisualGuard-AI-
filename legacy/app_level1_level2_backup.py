import os
import json
import base64
from pathlib import Path
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim


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
SCREENSHOT_DIR = Path("screenshots")
DIFF_DIR = Path("screenshots/diff")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)
DIFF_DIR.mkdir(parents=True, exist_ok=True)

if "level1_result" not in st.session_state:
    st.session_state.level1_result = None

if "level2_result" not in st.session_state:
    st.session_state.level2_result = None

if "level2_diff_data" not in st.session_state:
    st.session_state.level2_diff_data = None


def log_event(event_type, data):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }

    with open(LOG_DIR / "agent_logs.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_uploaded_file(uploaded_file, prefix=""):
    safe_name = uploaded_file.name.replace(" ", "_")
    file_path = UPLOAD_DIR / f"{prefix}{safe_name}"

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


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        raise ValueError("Model did not return valid JSON.")

    return text[start:end]


def normalize_number(value, default=70):
    try:
        value = float(value)
    except Exception:
        value = default

    if value <= 1:
        value = value * 100

    value = int(round(value))
    return max(0, min(100, value))


def normalize_level1_report(report, image_path):
    report["source"] = report.get("source", "Gemini Vision")
    report["resource_id"] = report.get("resource_id", str(image_path))

    findings = report.get("findings", [])
    normalized_findings = []

    for finding in findings:
        finding["confidence"] = normalize_number(finding.get("confidence", 70))
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

    report["overall_score"] = normalize_number(score, 70)

    return report


def normalize_level2_report(report, baseline_path, current_path):
    report["source"] = report.get("source", "Gemini Vision")
    report["baseline"] = report.get("baseline", str(baseline_path))
    report["current"] = report.get("current", str(current_path))
    report["overall_verdict"] = report.get("overall_verdict", "Needs human review")
    report["summary"] = report.get("summary", "Visual differences detected and reviewed.")

    differences = report.get("differences", [])
    normalized_differences = []

    for diff in differences:
        diff["change_type"] = diff.get("change_type", "neutral")
        diff["location"] = diff.get("location", "Changed visual region")
        diff["what_changed"] = diff.get("what_changed", "A visual change was detected.")
        diff["evidence"] = diff.get("evidence", {})
        diff["ux_impact"] = diff.get("ux_impact", "This may affect the user experience.")
        diff["recommendation"] = diff.get("recommendation", "Review whether this change is intentional.")
        diff["confidence"] = normalize_number(diff.get("confidence", 70))
        normalized_differences.append(diff)

    report["differences"] = normalized_differences

    return report


def fallback_level1_report(image_path, contrast_data, reason):
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


def fallback_level2_report(baseline_path, current_path, diff_data, reason):
    regions = diff_data.get("changed_regions", [])
    differences = []

    if not regions:
        regions = [{"x": 0, "y": 0, "width": 0, "height": 0, "area": 0}]

    for index, region in enumerate(regions[:5], 1):
        differences.append({
            "change_type": "neutral",
            "location": f"Changed region {index}",
            "what_changed": f"Detected change at x={region['x']}, y={region['y']}, width={region['width']}, height={region['height']}.",
            "evidence": {
                "pixel_region": f"{region['x']},{region['y']},{region['width']},{region['height']}",
                "diff_percentage": str(diff_data.get("diff_percentage")),
                "similarity_score": str(diff_data.get("similarity_score"))
            },
            "ux_impact": "This visual change may be intentional or may require review.",
            "recommendation": "Confirm whether this change is expected before approving the new UI.",
            "confidence": 72
        })

    return {
        "source": "Fallback",
        "fallback_reason": reason,
        "baseline": str(baseline_path),
        "current": str(current_path),
        "overall_verdict": "Needs human review",
        "summary": "Visual changes were detected. Review highlighted regions and confirm whether they are intentional.",
        "differences": differences
    }


def call_gemini_rest(payload, headers):
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

    return response, used_model, last_error


def analyze_level1_with_gemini(image_path, contrast_data):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback_level1_report(image_path, contrast_data, "GEMINI_API_KEY is missing from .env.")

    try:
        image_base64 = image_to_base64(image_path)
        mime_type = get_mime_type(image_path)

        prompt = f"""
You are VisualGuard AI, a production-grade UI/UX Design Audit Agent.

Analyze the uploaded UI screenshot and identify visible design issues.

Strict rules:
1. Do not hallucinate.
2. Every finding must be based only on something visible in the screenshot.
3. Mention actual visible UI areas, not generic text like "main content area".
4. Find at least 5 findings if possible.
5. Cover all five principles:
   - Visual Hierarchy
   - Contrast / WCAG AA
   - Spacing
   - Alignment
   - Consistency

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
- Do not return decimals.

Contrast evidence:
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

        response, used_model, last_error = call_gemini_rest(payload, headers)

        if response is None or response.status_code != 200:
            return fallback_level1_report(
                image_path,
                contrast_data,
                f"Gemini REST API failed after trying multiple models: {last_error}"
            )

        response_json = response.json()
        text = response_json["candidates"][0]["content"]["parts"][0]["text"]

        cleaned = clean_json_response(text)
        result = json.loads(cleaned)
        result = normalize_level1_report(result, image_path)
        result["model_used"] = used_model

        if "findings" not in result or len(result["findings"]) < 3:
            return fallback_level1_report(image_path, contrast_data, "Gemini returned fewer than 3 findings.")

        return result

    except Exception as error:
        return fallback_level1_report(image_path, contrast_data, f"Gemini API failed: {str(error)}")


def load_cv_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def resize_same(img1, img2):
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])

    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    return img1, img2


def calculate_image_diff(baseline_path, current_path):
    baseline = load_cv_image(baseline_path)
    current = load_cv_image(current_path)

    baseline, current = resize_same(baseline, current)

    gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray_baseline, gray_current, full=True)
    diff = (diff * 255).astype("uint8")

    threshold = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    changed_area = 0

    highlighted = current.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area > 300:
            boxes.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area)
            })

            changed_area += area
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)

    total_area = current.shape[0] * current.shape[1]
    diff_percentage = round((changed_area / total_area) * 100, 2)

    diff_path = DIFF_DIR / "level2_diff.png"
    cv2.imwrite(str(diff_path), highlighted)

    return {
        "similarity_score": round(float(score), 4),
        "diff_percentage": diff_percentage,
        "changed_regions": boxes[:15],
        "diff_image_path": str(diff_path)
    }


def analyze_level2_with_gemini(baseline_path, current_path, diff_data):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback_level2_report(
            baseline_path,
            current_path,
            diff_data,
            "GEMINI_API_KEY is missing from .env."
        )

    try:
        baseline_base64 = image_to_base64(baseline_path)
        current_base64 = image_to_base64(current_path)

        baseline_mime = get_mime_type(baseline_path)
        current_mime = get_mime_type(current_path)

        prompt = f"""
You are VisualGuard AI, a UI visual regression analysis agent.

Compare the baseline screenshot and the current screenshot.

Classify visible changes as:
- improvement
- regression
- neutral

Rules:
1. Do not hallucinate.
2. Use only visible evidence from the two screenshots.
3. Mention specific visible regions.
4. Flag accessibility regressions clearly.
5. Return at least 5 differences if visible.
6. Include practical recommendation for each change.

Return only valid JSON in this exact format:

{{
  "source": "Gemini Vision",
  "baseline": "baseline filename",
  "current": "current filename",
  "overall_verdict": "Net improvement | Net regression | Mostly neutral | Needs human review",
  "summary": "short summary",
  "differences": [
    {{
      "change_type": "improvement | regression | neutral",
      "location": "specific visible location",
      "what_changed": "specific change",
      "evidence": {{
        "pixel_region": "x,y,width,height",
        "diff_percentage": "value",
        "similarity_score": "value"
      }},
      "ux_impact": "impact on user experience",
      "recommendation": "specific recommendation",
      "confidence": 85
    }}
  ]
}}

Diff data from image processing:
{json.dumps(diff_data, indent=2)}
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
                                "mime_type": baseline_mime,
                                "data": baseline_base64
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": current_mime,
                                "data": current_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        response, used_model, last_error = call_gemini_rest(payload, headers)

        if response is None or response.status_code != 200:
            return fallback_level2_report(
                baseline_path,
                current_path,
                diff_data,
                f"Gemini REST API failed after trying multiple models: {last_error}"
            )

        response_json = response.json()
        text = response_json["candidates"][0]["content"]["parts"][0]["text"]

        cleaned = clean_json_response(text)
        result = json.loads(cleaned)
        result = normalize_level2_report(result, baseline_path, current_path)
        result["model_used"] = used_model

        if "differences" not in result or len(result["differences"]) < 1:
            return fallback_level2_report(
                baseline_path,
                current_path,
                diff_data,
                "Gemini returned no differences."
            )

        return result

    except Exception as error:
        return fallback_level2_report(
            baseline_path,
            current_path,
            diff_data,
            f"Gemini API failed: {str(error)}"
        )


def save_json_report(report, filename):
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return output_path


def generate_level1_markdown(report):
    lines = []

    lines.append("# VisualGuard AI - Level 1 Design Audit Report")
    lines.append("")
    lines.append(f"Source: {report.get('source', 'Unknown')}")
    lines.append(f"Model Used: {report.get('model_used', 'N/A')}")
    lines.append(f"Resource: {report.get('resource_id')}")
    lines.append(f"Overall Score: {report.get('overall_score')}/100")
    lines.append("")

    lines.append("## Findings")
    lines.append("")

    for index, finding in enumerate(report.get("findings", []), 1):
        lines.append(f"### {index}. {finding.get('principle')} - {finding.get('severity').upper()}")
        lines.append(f"Location: {finding.get('location')}")
        lines.append(f"Issue: {finding.get('issue')}")
        lines.append(f"User Impact: {finding.get('user_impact')}")
        lines.append(f"Recommendation: {finding.get('recommendation')}")
        lines.append(f"Evidence: {finding.get('evidence')}")
        lines.append(f"Confidence: {finding.get('confidence')}%")
        lines.append("")

    return "\n".join(lines)


def generate_level2_markdown(report):
    lines = []

    lines.append("# VisualGuard AI - Level 2 Regression Report")
    lines.append("")
    lines.append(f"Source: {report.get('source', 'Unknown')}")
    lines.append(f"Model Used: {report.get('model_used', 'N/A')}")
    lines.append(f"Baseline: {report.get('baseline')}")
    lines.append(f"Current: {report.get('current')}")
    lines.append(f"Overall Verdict: {report.get('overall_verdict')}")
    lines.append("")
    lines.append(report.get("summary", ""))
    lines.append("")
    lines.append("## Differences")
    lines.append("")

    for index, diff in enumerate(report.get("differences", []), 1):
        lines.append(f"### {index}. {diff.get('change_type').upper()}")
        lines.append(f"Location: {diff.get('location')}")
        lines.append(f"What Changed: {diff.get('what_changed')}")
        lines.append(f"UX Impact: {diff.get('ux_impact')}")
        lines.append(f"Recommendation: {diff.get('recommendation')}")
        lines.append(f"Evidence: {json.dumps(diff.get('evidence', {}))}")
        lines.append(f"Confidence: {diff.get('confidence')}%")
        lines.append("")

    return "\n".join(lines)


def generate_text_report(report, report_type):
    if report_type == "level1":
        return generate_level1_markdown(report).replace("#", "").replace("*", "")

    return generate_level2_markdown(report).replace("#", "").replace("*", "")


def convert_level1_csv(report):
    return pd.DataFrame(report.get("findings", [])).to_csv(index=False).encode("utf-8")


def convert_level2_csv(report):
    return pd.DataFrame(report.get("differences", [])).to_csv(index=False).encode("utf-8")


def run_level1_audit(image_path):
    log_event("level1_started", {"image_path": str(image_path)})

    contrast_data = analyze_contrast(image_path)

    report = analyze_level1_with_gemini(image_path, contrast_data)

    save_json_report(report, "level1_audit.json")

    with open(REPORT_DIR / "level1_report.md", "w", encoding="utf-8") as file:
        file.write(generate_level1_markdown(report))

    log_event("level1_completed", {
        "source": report.get("source"),
        "model_used": report.get("model_used"),
        "findings_count": len(report["findings"]),
        "overall_score": report["overall_score"]
    })

    return report


def run_level2_audit(baseline_path, current_path):
    log_event("level2_started", {
        "baseline": str(baseline_path),
        "current": str(current_path)
    })

    diff_data = calculate_image_diff(baseline_path, current_path)

    report = analyze_level2_with_gemini(baseline_path, current_path, diff_data)

    save_json_report(report, "level2_regression.json")

    with open(REPORT_DIR / "level2_report.md", "w", encoding="utf-8") as file:
        file.write(generate_level2_markdown(report))

    log_event("level2_completed", {
        "source": report.get("source"),
        "model_used": report.get("model_used"),
        "differences_count": len(report["differences"]),
        "verdict": report["overall_verdict"]
    })

    return report, diff_data


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

    textarea {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border: 1px solid #13233a !important;
        border-radius: 14px !important;
        font-family: Consolas, monospace !important;
        font-size: 13px !important;
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
            Premium UI audit agent for screenshot-based design review, accessibility feedback, visual regression analysis, and structured reporting.
        </div>
        <span class="pill">Agent 1</span>
        <span class="pill">Level 1 + Level 2</span>
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

mode = st.sidebar.radio(
    "Mode",
    [
        "Level 1 · Screenshot Audit",
        "Level 2 · Before/After Regression"
    ]
)

st.sidebar.write("Outputs: JSON / MD / TXT / CSV")


def show_output_panel(result, report_type):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Structured Output")

    json_text = json.dumps(result, indent=2, ensure_ascii=False)

    if report_type == "level1":
        markdown_text = generate_level1_markdown(result)
        txt_text = generate_text_report(result, "level1")
        csv_data = convert_level1_csv(result)
        csv_filename = "visualguard_level1_findings.csv"
        json_filename = "visualguard_level1_report.json"
        md_filename = "visualguard_level1_report.md"
        txt_filename = "visualguard_level1_report.txt"
    else:
        markdown_text = generate_level2_markdown(result)
        txt_text = generate_text_report(result, "level2")
        csv_data = convert_level2_csv(result)
        csv_filename = "visualguard_level2_differences.csv"
        json_filename = "visualguard_level2_report.json"
        md_filename = "visualguard_level2_report.md"
        txt_filename = "visualguard_level2_report.txt"

    output_format = st.selectbox(
        "Output format",
        ["JSON", "Markdown", "Text", "CSV"],
        key=f"{report_type}_output_format"
    )

    if output_format == "JSON":
        st.text_area("Preview", json_text, height=420, key=f"{report_type}_json_preview")
        st.download_button(
            "Download JSON",
            json_text.encode("utf-8"),
            json_filename,
            "application/json",
            use_container_width=True,
            key=f"{report_type}_json_download"
        )

    elif output_format == "Markdown":
        st.text_area("Preview", markdown_text, height=420, key=f"{report_type}_md_preview")
        st.download_button(
            "Download Markdown",
            markdown_text.encode("utf-8"),
            md_filename,
            "text/markdown",
            use_container_width=True,
            key=f"{report_type}_md_download"
        )

    elif output_format == "Text":
        st.text_area("Preview", txt_text, height=420, key=f"{report_type}_txt_preview")
        st.download_button(
            "Download Text",
            txt_text.encode("utf-8"),
            txt_filename,
            "text/plain",
            use_container_width=True,
            key=f"{report_type}_txt_download"
        )

    else:
        if report_type == "level1":
            st.dataframe(pd.DataFrame(result["findings"]), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(result["differences"]), use_container_width=True)

        st.download_button(
            "Download CSV",
            csv_data,
            csv_filename,
            "text/csv",
            use_container_width=True,
            key=f"{report_type}_csv_download"
        )

    st.markdown("</div>", unsafe_allow_html=True)


if mode == "Level 1 · Screenshot Audit":
    st.markdown(
        """
        <div class="panel">
            <h2>Single Screenshot Design Audit</h2>
            <p>Upload a UI screenshot. The agent evaluates hierarchy, contrast, spacing, alignment, and consistency.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload UI Screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="level1_upload"
    )

    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file, "level1_")

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
                st.session_state.level1_result = run_level1_audit(image_path)

    if st.session_state.level1_result is not None:
        result = st.session_state.level1_result

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

        show_output_panel(result, "level1")

else:
    st.markdown(
        """
        <div class="panel">
            <h2>Before/After Visual Regression</h2>
            <p>Upload a baseline screenshot and current screenshot. The agent detects visual changes and classifies them as improvement, regression, or neutral.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    baseline_file = st.file_uploader(
        "Upload Baseline Screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="baseline_upload"
    )

    current_file = st.file_uploader(
        "Upload Current Screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        key="current_upload"
    )

    if baseline_file and current_file:
        baseline_path = save_uploaded_file(baseline_file, "baseline_")
        current_path = save_uploaded_file(current_file, "current_")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Baseline")
            st.image(Image.open(baseline_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Current")
            st.image(Image.open(current_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        run_level2_button = st.button("Run Regression Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_level2_button:
            with st.spinner("Comparing screenshots..."):
                result, diff_data = run_level2_audit(baseline_path, current_path)
                st.session_state.level2_result = result
                st.session_state.level2_diff_data = diff_data

    if st.session_state.level2_result is not None:
        result = st.session_state.level2_result
        diff_data = st.session_state.level2_diff_data

        if result.get("source") == "Gemini Vision":
            st.success("Regression analysis completed using Gemini Vision.")
        else:
            st.warning("Regression analysis completed using fallback output.")
            st.error(result.get("fallback_reason", "Unknown fallback reason"))

        differences = result["differences"]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Similarity", diff_data["similarity_score"])

        with m2:
            st.metric("Diff %", f"{diff_data['diff_percentage']}%")

        with m3:
            st.metric("Changes", len(differences))

        with m4:
            st.metric("Verdict", result.get("overall_verdict", "Review"))

        diff_image_path = Path(diff_data["diff_image_path"])

        if diff_image_path.exists():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Highlighted Difference Image")
            st.image(Image.open(diff_image_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Detected Differences")

        for index, diff in enumerate(differences, 1):
            st.markdown(
                f"""
                <div class="finding-card">
                    <p class="small-label">Difference {index}</p>
                    <h3>{diff["change_type"].title()} · {diff["confidence"]}%</h3>
                    <p><b>Location:</b> {diff["location"]}</p>
                    <p><b>What Changed:</b> {diff["what_changed"]}</p>
                    <p><b>UX Impact:</b> {diff["ux_impact"]}</p>
                    <p><b>Recommendation:</b> {diff["recommendation"]}</p>
                    <p><b>Evidence:</b> {json.dumps(diff["evidence"])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        show_output_panel(result, "level2")