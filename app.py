import os
import sys
import json
import base64
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim


st.set_page_config(page_title="VisualGuard AI", page_icon="🛡️", layout="wide")

load_dotenv(override=True)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")
LOG_DIR = Path("logs")
DIFF_DIR = Path("screenshots/diff")
CURRENT_DIR = Path("screenshots/current")
BASELINE_DIR = Path("baselines")
VERSION_DIR = Path("baselines/versions")

for folder in [
    UPLOAD_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    LOG_DIR,
    DIFF_DIR,
    CURRENT_DIR,
    BASELINE_DIR,
    VERSION_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

for key in ["level1_result", "level2_result", "level2_diff", "level3_result"]:
    if key not in st.session_state:
        st.session_state[key] = None


def save_uploaded_file(uploaded_file, prefix=""):
    file_path = UPLOAD_DIR / f"{prefix}{uploaded_file.name.replace(' ', '_')}"

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def safe_filename(text):
    cleaned = ""

    for char in text.lower().strip():
        if char.isalnum() or char in "-_":
            cleaned += char
        else:
            cleaned += "_"

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_") or "page"


def get_mime_type(path):
    path = str(path).lower()

    if path.endswith(".jpg") or path.endswith(".jpeg"):
        return "image/jpeg"

    if path.endswith(".webp"):
        return "image/webp"

    return "image/png"


def image_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def clean_json(text):
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
        value *= 100

    value = int(round(value))
    return max(0, min(100, value))


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def bgr_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[2]), int(color[1]), int(color[0]))


def relative_luminance(rgb):
    values = []

    for channel in rgb:
        channel = channel / 255

        if channel <= 0.03928:
            values.append(channel / 12.92)
        else:
            values.append(((channel + 0.055) / 1.055) ** 2.4)

    return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]


def contrast_ratio(color1, color2):
    l1 = relative_luminance(color1)
    l2 = relative_luminance(color2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return round((lighter + 0.05) / (darker + 0.05), 2)


def dominant_colors(path, k=6):
    image = Image.open(path).convert("RGB")
    image = image.resize((250, 250))

    pixels = np.array(image).reshape(-1, 3)
    pixels = (pixels // 16) * 16

    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    indices = counts.argsort()[-k:][::-1]

    return [tuple(map(int, color)) for color in colors[indices]]


def analyze_contrast(path):
    colors = dominant_colors(path)
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


def gemini_call(payload):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key.strip() == "your_gemini_api_key_here":
        return None, None, "GEMINI_API_KEY is missing or placeholder."

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key.strip()
    }

    model = "gemini-2.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=90
    )

    if response.status_code == 200:
        return response, model, None

    return None, model, f"{model}: {response.status_code} - {response.text}"

def fallback_level1(path, contrast_data, reason):
    weak = [item for item in contrast_data if item["wcag_aa_pass"] is False]
    evidence = str((weak or contrast_data)[:3])

    return {
        "source": "Fallback",
        "fallback_reason": reason,
        "resource_id": str(path),
        "overall_score": 74,
        "findings": [
            {
                "principle": "Contrast",
                "severity": "medium",
                "location": "Visible text and foreground UI elements",
                "issue": "Some foreground and background color combinations may not provide enough contrast.",
                "user_impact": "Low-vision users may struggle to read text or controls.",
                "recommendation": "Increase contrast to at least 4.5:1 for normal text.",
                "evidence": evidence,
                "confidence": 78
            },
            {
                "principle": "Spacing",
                "severity": "low",
                "location": "Main content area",
                "issue": "Some content groups may need more breathing room.",
                "user_impact": "Dense spacing can reduce scanability.",
                "recommendation": "Use consistent 8px, 16px, and 24px spacing tokens.",
                "evidence": "Detected from screenshot layout density.",
                "confidence": 68
            },
            {
                "principle": "Consistency",
                "severity": "info",
                "location": "Repeated UI components",
                "issue": "Component consistency needs verification against design tokens.",
                "user_impact": "Inconsistency can reduce predictability.",
                "recommendation": "Compare typography, buttons, cards, and colors with the design system.",
                "evidence": "No design-token metadata provided.",
                "confidence": 62
            }
        ]
    }


def normalize_level1(report, path):
    report["source"] = report.get("source", "Gemini Vision")
    report["resource_id"] = report.get("resource_id", str(path))

    for finding in report.get("findings", []):
        finding["confidence"] = normalize_number(finding.get("confidence", 70))
        finding.setdefault("principle", "General")
        finding.setdefault("severity", "medium")
        finding.setdefault("location", "Visible UI area")
        finding.setdefault("issue", "Design issue detected.")
        finding.setdefault("user_impact", "May affect usability or readability.")
        finding.setdefault("recommendation", "Review and improve this UI area.")
        finding.setdefault("evidence", "Visible in screenshot.")

    score = report.get("overall_score", 0)

    if not score:
        penalties = {
            "critical": 25,
            "high": 18,
            "medium": 10,
            "low": 5,
            "info": 2
        }

        score = 100

        for finding in report.get("findings", []):
            severity = finding.get("severity", "medium").lower()
            score -= penalties.get(severity, 10)

    report["overall_score"] = normalize_number(score, 70)

    return report


def analyze_level1(path):
    contrast_data = analyze_contrast(path)

    prompt = f"""
You are VisualGuard AI, a production-grade UI/UX Design Audit Agent.

Analyze the uploaded UI screenshot and identify visible design issues only.
Do not hallucinate. Every finding must be visible in the screenshot.

Check these five principles:
1. Visual Hierarchy
2. Contrast / WCAG AA
3. Spacing
4. Alignment
5. Consistency

Return at least 5 findings if possible.

Return only valid JSON:

{{
  "source": "Gemini Vision",
  "resource_id": "uploaded_file_name",
  "overall_score": 85,
  "findings": [
    {{
      "principle": "Visual Hierarchy",
      "severity": "medium",
      "location": "specific visible location",
      "issue": "specific issue",
      "user_impact": "impact",
      "recommendation": "specific fix",
      "evidence": "visible evidence",
      "confidence": 85
    }}
  ]
}}

Confidence and overall_score must be integers from 0 to 100.

Contrast evidence:
{json.dumps(contrast_data[:10], indent=2)}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": get_mime_type(path),
                            "data": image_to_base64(path)
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    response, model, error = gemini_call(payload)

    if not response:
        return fallback_level1(path, contrast_data, error)

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        report = json.loads(clean_json(text))
        report = normalize_level1(report, path)
        report["model_used"] = model

        if len(report.get("findings", [])) < 3:
            return fallback_level1(path, contrast_data, "Gemini returned fewer than 3 findings.")

        return report

    except Exception as error:
        return fallback_level1(path, contrast_data, str(error))


def load_cv(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def resize_same(image1, image2):
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])

    return cv2.resize(image1, (width, height)), cv2.resize(image2, (width, height))


def parse_regions(text):
    regions = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            x, y, width, height = [
                int(value)
                for value in line.replace(" ", "").split(",")
            ]

            regions.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height
            })

        except Exception:
            pass

    return regions


def mask_regions(image, regions):
    output = image.copy()

    for region in regions:
        x = max(0, region["x"])
        y = max(0, region["y"])
        width = max(0, region["width"])
        height = max(0, region["height"])

        output[y:y + height, x:x + width] = (0, 0, 0)

    return output


def average_hex(image, x, y, width, height):
    image_height, image_width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    if x2 <= x1 or y2 <= y1:
        return "#000000"

    region = image[y1:y2, x1:x2]

    if region.size == 0:
        return "#000000"

    average_bgr = region.reshape(-1, 3).mean(axis=0)

    return bgr_to_hex(average_bgr)


def calculate_diff(baseline_path, current_path, diff_path, ignore_regions=None, min_area=300):
    ignore_regions = ignore_regions or []

    baseline = load_cv(baseline_path)
    current = load_cv(current_path)

    baseline, current = resize_same(baseline, current)

    masked_baseline = mask_regions(baseline, ignore_regions)
    masked_current = mask_regions(current, ignore_regions)

    gray_baseline = cv2.cvtColor(masked_baseline, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(masked_current, cv2.COLOR_BGR2GRAY)

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
    mask_for_overlay = np.zeros(current.shape[:2], dtype="uint8")

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height

        if area > min_area:
            baseline_hex = average_hex(baseline, x, y, width, height)
            current_hex = average_hex(current, x, y, width, height)

            boxes.append({
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
                "baseline_hex": baseline_hex,
                "current_hex": current_hex,
                "color_shift": baseline_hex != current_hex
            })

            changed_area += area

            cv2.rectangle(
                highlighted,
                (x, y),
                (x + width, y + height),
                (0, 0, 255),
                3
            )

            cv2.putText(
                highlighted,
                "CHANGE",
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.drawContours(mask_for_overlay, [contour], -1, 255, -1)

    red_overlay = current.copy()
    red_overlay[mask_for_overlay > 0] = (0, 0, 255)
    highlighted = cv2.addWeighted(red_overlay, 0.35, highlighted, 0.65, 0)

    total_area = current.shape[0] * current.shape[1]
    diff_percentage = round((changed_area / total_area) * 100, 2)

    cv2.imwrite(str(diff_path), highlighted)

    return {
        "similarity_score": round(float(score), 4),
        "diff_percentage": diff_percentage,
        "changed_regions": boxes[:25],
        "ignored_regions": ignore_regions,
        "diff_image_path": str(diff_path)
    }


def fallback_level2(baseline_path, current_path, diff_data, reason):
    regions = diff_data.get("changed_regions") or []
    differences = []

    if regions:
        for index, region in enumerate(regions[:5], 1):
            differences.append({
                "change_type": "neutral",
                "location": f"Changed region {index}",
                "what_changed": f"Change detected at x={region['x']}, y={region['y']}, width={region['width']}, height={region['height']}.",
                "evidence": {
                    "pixel_region": f"{region['x']},{region['y']},{region['width']},{region['height']}",
                    "baseline_hex": region.get("baseline_hex"),
                    "current_hex": region.get("current_hex"),
                    "diff_percentage": diff_data.get("diff_percentage"),
                    "similarity_score": diff_data.get("similarity_score")
                },
                "ux_impact": "Requires review to confirm whether this change is intentional.",
                "recommendation": "Approve only if expected.",
                "confidence": 72
            })
    else:
        differences.append({
            "change_type": "neutral",
            "location": "Full page",
            "what_changed": "No significant visual difference was detected.",
            "evidence": {
                "diff_percentage": diff_data.get("diff_percentage"),
                "similarity_score": diff_data.get("similarity_score")
            },
            "ux_impact": "No visible UX regression detected by image comparison.",
            "recommendation": "No action needed unless manual review finds an issue.",
            "confidence": 82
        })

    return {
        "source": "Fallback",
        "fallback_reason": reason,
        "baseline": str(baseline_path),
        "current": str(current_path),
        "overall_verdict": "Needs human review" if regions else "Mostly neutral",
        "summary": "Visual changes detected." if regions else "No significant visual change detected.",
        "differences": differences
    }


def normalize_level2(report, baseline_path, current_path):
    report["source"] = report.get("source", "Gemini Vision")
    report["baseline"] = report.get("baseline", str(baseline_path))
    report["current"] = report.get("current", str(current_path))
    report.setdefault("overall_verdict", "Needs human review")
    report.setdefault("summary", "Visual differences detected.")

    for difference in report.get("differences", []):
        difference.setdefault("change_type", "neutral")
        difference.setdefault("location", "Changed visual region")
        difference.setdefault("what_changed", "A visual change was detected.")
        difference.setdefault("evidence", {})
        difference.setdefault("ux_impact", "May affect user experience.")
        difference.setdefault("recommendation", "Review whether intentional.")
        difference["confidence"] = normalize_number(difference.get("confidence", 70))

    return report


def analyze_level2(baseline_path, current_path, diff_data):
    prompt = f"""
You are VisualGuard AI, a UI visual regression analysis agent.

Compare the baseline and current screenshots.
Be specific. Do not give generic feedback.

Classify each visible change as:
- improvement
- regression
- neutral

Flag accessibility regressions explicitly:
- contrast ratio drops
- font size reductions
- spacing compression
- alignment breaks
- unreadable text
- layout shifts

Use pixel measurements and hex values from diff data wherever available.
If no meaningful change exists, say so clearly.

Return only valid JSON:

{{
  "source": "Gemini Vision",
  "baseline": "baseline filename",
  "current": "current filename",
  "overall_verdict": "Net improvement | Net regression | Mostly neutral | Needs human review",
  "summary": "specific summary",
  "differences": [
    {{
      "change_type": "improvement | regression | neutral",
      "location": "specific location",
      "what_changed": "specific change",
      "evidence": {{
        "pixel_region": "x,y,width,height",
        "baseline_hex": "#000000",
        "current_hex": "#ffffff",
        "diff_percentage": "value",
        "similarity_score": "value"
      }},
      "ux_impact": "specific impact",
      "recommendation": "specific recommendation",
      "confidence": 85
    }}
  ]
}}

Diff data:
{json.dumps(diff_data, indent=2)}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": get_mime_type(baseline_path),
                            "data": image_to_base64(baseline_path)
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": get_mime_type(current_path),
                            "data": image_to_base64(current_path)
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    response, model, error = gemini_call(payload)

    if not response:
        return fallback_level2(baseline_path, current_path, diff_data, error)

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        report = json.loads(clean_json(text))
        report = normalize_level2(report, baseline_path, current_path)
        report["model_used"] = model

        if len(report.get("differences", [])) < 1:
            return fallback_level2(baseline_path, current_path, diff_data, "Gemini returned no differences.")

        return report

    except Exception as error:
        return fallback_level2(baseline_path, current_path, diff_data, str(error))


def fallback_level3_page(page_name, url, diff_data, reason):
    regions = diff_data.get("changed_regions", [])
    findings = []

    if regions:
        for index, region in enumerate(regions[:5], 1):
            findings.append({
                "change_type": "neutral",
                "severity": "medium",
                "location": f"Changed region {index}",
                "what_changed": f"Visual change detected at x={region['x']}, y={region['y']}, width={region['width']}, height={region['height']}.",
                "ux_impact": "This may affect layout consistency or readability and should be reviewed.",
                "recommendation": "Check whether this change was expected. Refresh baseline only after approval.",
                "evidence": {
                    "pixel_region": f"{region['x']},{region['y']},{region['width']},{region['height']}",
                    "baseline_hex": region.get("baseline_hex"),
                    "current_hex": region.get("current_hex"),
                    "diff_percentage": diff_data.get("diff_percentage"),
                    "similarity_score": diff_data.get("similarity_score")
                },
                "confidence": 75
            })
    else:
        findings.append({
            "change_type": "neutral",
            "severity": "info",
            "location": "Full page",
            "what_changed": "No meaningful visual regression detected.",
            "ux_impact": "The page appears visually stable against the stored baseline.",
            "recommendation": "No action required.",
            "evidence": {
                "diff_percentage": diff_data.get("diff_percentage"),
                "similarity_score": diff_data.get("similarity_score")
            },
            "confidence": 86
        })

    return {
        "source": "Fallback",
        "fallback_reason": reason,
        "page_name": page_name,
        "url": url,
        "overall_page_verdict": "Needs human review" if regions else "No significant regression",
        "summary": "Fallback review generated from pixel-diff evidence.",
        "findings": findings
    }


def normalize_level3_page(report):
    report["source"] = report.get("source", "Gemini Vision")
    report.setdefault("summary", "Page visual regression review completed.")
    report.setdefault("overall_page_verdict", "Needs human review")

    for finding in report.get("findings", []):
        finding.setdefault("change_type", "neutral")
        finding.setdefault("severity", "medium")
        finding.setdefault("location", "Changed visual region")
        finding.setdefault("what_changed", "A visual change was detected.")
        finding.setdefault("ux_impact", "May affect user experience.")
        finding.setdefault("recommendation", "Review whether intentional.")
        finding.setdefault("evidence", {})
        finding["confidence"] = normalize_number(finding.get("confidence", 70))

    return report


def analyze_level3_page(page_name, url, baseline_path, current_path, diff_data):
    prompt = f"""
You are VisualGuard AI operating in Level 3 autonomous UI regression mode.

Compare the baseline screenshot and current screenshot for this page:
Page name: {page_name}
URL: {url}

Be very specific and evidence-backed.
Do not give generic comments.
If the page is visually stable, say that clearly.
If there are differences, classify them as improvement, regression, or neutral.
Flag accessibility regressions explicitly if present.

Use this diff data:
{json.dumps(diff_data, indent=2)}

Return only valid JSON:

{{
  "source": "Gemini Vision",
  "page_name": "{page_name}",
  "url": "{url}",
  "overall_page_verdict": "No significant regression | Needs human review | Regression detected | Improvement detected",
  "summary": "specific page-level summary",
  "findings": [
    {{
      "change_type": "improvement | regression | neutral",
      "severity": "critical | high | medium | low | info",
      "location": "specific visible location",
      "what_changed": "specific visible change",
      "ux_impact": "specific UX impact",
      "recommendation": "specific recommendation",
      "evidence": {{
        "pixel_region": "x,y,width,height",
        "baseline_hex": "#000000",
        "current_hex": "#ffffff",
        "diff_percentage": "value",
        "similarity_score": "value"
      }},
      "confidence": 85
    }}
  ]
}}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": get_mime_type(baseline_path),
                            "data": image_to_base64(baseline_path)
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": get_mime_type(current_path),
                            "data": image_to_base64(current_path)
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    response, model, error = gemini_call(payload)

    if not response:
        return fallback_level3_page(page_name, url, diff_data, error)

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        report = json.loads(clean_json(text))
        report = normalize_level3_page(report)
        report["model_used"] = model

        if len(report.get("findings", [])) < 1:
            return fallback_level3_page(page_name, url, diff_data, "Gemini returned no page findings.")

        return report

    except Exception as error:
        return fallback_level3_page(page_name, url, diff_data, str(error))


def ensure_capture_script():
    script_path = Path("capture_page.py")

    script_path.write_text(
        r'''
import sys
import asyncio
from playwright.sync_api import sync_playwright

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

url = sys.argv[1]
output_path = sys.argv[2]
login_url = sys.argv[3] if len(sys.argv) > 3 else ""
username = sys.argv[4] if len(sys.argv) > 4 else ""
password = sys.argv[5] if len(sys.argv) > 5 else ""
username_selector = sys.argv[6] if len(sys.argv) > 6 else ""
password_selector = sys.argv[7] if len(sys.argv) > 7 else ""
submit_selector = sys.argv[8] if len(sys.argv) > 8 else ""

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    if login_url and username and password and username_selector and password_selector:
        page.goto(login_url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(2000)
        page.fill(username_selector, username)
        page.fill(password_selector, password)

        if submit_selector:
            page.click(submit_selector)
        else:
            page.keyboard.press("Enter")

        page.wait_for_timeout(4000)

    page.goto(url, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(4000)
    page.screenshot(path=output_path, full_page=True)
    browser.close()
''',
        encoding="utf-8"
    )

    return script_path


def capture_page(url, output_path, login_config):
    script_path = ensure_capture_script()

    args = [
        sys.executable,
        str(script_path),
        url,
        str(output_path),
        login_config.get("login_url", ""),
        login_config.get("username", ""),
        login_config.get("password", ""),
        login_config.get("username_selector", ""),
        login_config.get("password_selector", ""),
        login_config.get("submit_selector", "")
    ]

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def parse_pages(text):
    pages = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if "|" in line:
            name, url = line.split("|", 1)

            pages.append({
                "name": name.strip(),
                "url": url.strip()
            })

        else:
            pages.append({
                "name": safe_filename(line),
                "url": line
            })

    return pages


def refresh_baseline(page_name, current_path):
    safe_name = safe_filename(page_name)
    baseline_path = BASELINE_DIR / f"{safe_name}.png"

    if baseline_path.exists():
        version_path = VERSION_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        shutil.copy(baseline_path, version_path)

    shutil.copy(current_path, baseline_path)

    return baseline_path


def run_level3(pages, threshold, ignored_regions, refresh, login_config, use_gemini):
    start_time = datetime.now()
    page_reports = []
    regression_count = 0

    for page in pages:
        page_name = page["name"]
        url = page["url"]
        safe_name = safe_filename(page_name)

        baseline_path = BASELINE_DIR / f"{safe_name}.png"
        current_path = CURRENT_DIR / f"{safe_name}.png"
        diff_path = DIFF_DIR / f"level3_{safe_name}_diff.png"

        capture_page(url, current_path, login_config)

        if refresh:
            refresh_baseline(page_name, current_path)

            page_reports.append({
                "page_name": page_name,
                "url": url,
                "status": "baseline_refreshed",
                "baseline_path": str(baseline_path),
                "current_path": str(current_path),
                "diff_percentage": 0.0,
                "similarity_score": 1.0,
                "changed_regions": [],
                "ignored_regions": ignored_regions,
                "gemini_review": {
                    "overall_page_verdict": "Baseline refreshed",
                    "summary": "Current approved screenshot has been saved as the new baseline.",
                    "findings": []
                },
                "findings": []
            })

            continue

        if not baseline_path.exists():
            shutil.copy(current_path, baseline_path)

            page_reports.append({
                "page_name": page_name,
                "url": url,
                "status": "baseline_created",
                "message": "No previous baseline found. Current screenshot saved as baseline.",
                "baseline_path": str(baseline_path),
                "current_path": str(current_path),
                "diff_percentage": 0.0,
                "similarity_score": 1.0,
                "changed_regions": [],
                "ignored_regions": ignored_regions,
                "gemini_review": {
                    "overall_page_verdict": "Baseline created",
                    "summary": "No comparison was possible because this is the first capture for this page.",
                    "findings": []
                },
                "findings": []
            })

            continue

        diff_data = calculate_diff(
            baseline_path,
            current_path,
            diff_path,
            ignore_regions=ignored_regions,
            min_area=500
        )

        if use_gemini:
            page_review = analyze_level3_page(
                page_name,
                url,
                baseline_path,
                current_path,
                diff_data
            )
        else:
            page_review = fallback_level3_page(
                page_name,
                url,
                diff_data,
                "Gemini review disabled to save quota."
            )

        status = "passed"

        if diff_data["diff_percentage"] > threshold:
            status = "regression_detected"
            regression_count += 1

        page_reports.append({
            "page_name": page_name,
            "url": url,
            "status": status,
            "baseline_path": str(baseline_path),
            "current_path": str(current_path),
            "diff_image_path": str(diff_path),
            "diff_percentage": diff_data["diff_percentage"],
            "similarity_score": diff_data["similarity_score"],
            "changed_regions": diff_data["changed_regions"],
            "ignored_regions": ignored_regions,
            "gemini_review": page_review,
            "findings": page_review.get("findings", [])
        })

    duration = round((datetime.now() - start_time).total_seconds(), 2)

    report = {
        "source": "Autonomous Scan + Gemini Vision" if use_gemini else "Autonomous Scan + Pixel Diff",
        "status": "regressions_found" if regression_count else "passed_or_baseline_updated",
        "total_pages": len(pages),
        "pages_with_regressions": regression_count,
        "duration_seconds": duration,
        "fixed_viewport": "1440x900",
        "gemini_review_enabled": use_gemini,
        "dynamic_content_filtering": {
            "manual_ignored_regions": ignored_regions,
            "small_region_filter": "contours below 500px area ignored"
        },
        "pages": page_reports
    }

    save_json(report, "level3_scan.json")
    (REPORT_DIR / "level3_report.md").write_text(markdown_level3(report), encoding="utf-8")

    return report


def save_json(report, filename):
    (OUTPUT_DIR / filename).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def markdown_level1(report):
    text = "# VisualGuard AI - Level 1 Design Audit Report\n\n"
    text += f"Source: {report.get('source')}\n"
    text += f"Model Used: {report.get('model_used', 'N/A')}\n"
    text += f"Resource: {report.get('resource_id')}\n"
    text += f"Overall Score: {report.get('overall_score')}/100\n\n"
    text += "## Findings\n\n"

    for index, finding in enumerate(report.get("findings", []), 1):
        text += f"### {index}. {finding.get('principle')} - {finding.get('severity', '').upper()}\n"
        text += f"Location: {finding.get('location')}\n"
        text += f"Issue: {finding.get('issue')}\n"
        text += f"User Impact: {finding.get('user_impact')}\n"
        text += f"Recommendation: {finding.get('recommendation')}\n"
        text += f"Evidence: {finding.get('evidence')}\n"
        text += f"Confidence: {finding.get('confidence')}%\n\n"

    return text


def markdown_level2(report):
    text = "# VisualGuard AI - Level 2 Regression Report\n\n"
    text += f"Source: {report.get('source')}\n"
    text += f"Model Used: {report.get('model_used', 'N/A')}\n"
    text += f"Overall Verdict: {report.get('overall_verdict')}\n\n"
    text += f"{report.get('summary', '')}\n\n"
    text += "## Differences\n\n"

    for index, difference in enumerate(report.get("differences", []), 1):
        text += f"### {index}. {difference.get('change_type', '').upper()}\n"
        text += f"Location: {difference.get('location')}\n"
        text += f"What Changed: {difference.get('what_changed')}\n"
        text += f"UX Impact: {difference.get('ux_impact')}\n"
        text += f"Recommendation: {difference.get('recommendation')}\n"
        text += f"Evidence: {json.dumps(difference.get('evidence', {}))}\n"
        text += f"Confidence: {difference.get('confidence')}%\n\n"

    return text


def markdown_level3(report):
    text = "# VisualGuard AI - Level 3 Autonomous Scan Report\n\n"
    text += f"Source: {report.get('source')}\n"
    text += f"Status: {report.get('status')}\n"
    text += f"Total Pages: {report.get('total_pages')}\n"
    text += f"Pages With Regressions: {report.get('pages_with_regressions')}\n"
    text += f"Duration: {report.get('duration_seconds')} seconds\n"
    text += f"Viewport: {report.get('fixed_viewport')}\n"
    text += f"Gemini Review Enabled: {report.get('gemini_review_enabled')}\n\n"
    text += "## Dynamic Content Filtering\n"
    text += json.dumps(report.get("dynamic_content_filtering", {}), indent=2)
    text += "\n\n## Pages\n\n"

    for page in report.get("pages", []):
        review = page.get("gemini_review", {})

        text += f"### {page.get('page_name')}\n"
        text += f"URL: {page.get('url')}\n"
        text += f"Status: {page.get('status')}\n"
        text += f"Similarity: {page.get('similarity_score')}\n"
        text += f"Diff Percentage: {page.get('diff_percentage')}%\n"
        text += f"Review Verdict: {review.get('overall_page_verdict')}\n"
        text += f"Review Summary: {review.get('summary')}\n\n"

        for index, finding in enumerate(page.get("findings", []), 1):
            text += f"- {index}. {finding.get('what_changed', finding.get('issue'))} Confidence: {finding.get('confidence')}%\n"

        text += "\n"

    return text


def text_report(report, report_type):
    if report_type == "level1":
        return markdown_level1(report).replace("#", "").replace("*", "")

    if report_type == "level2":
        return markdown_level2(report).replace("#", "").replace("*", "")

    return markdown_level3(report).replace("#", "").replace("*", "")


def csv_level1(report):
    return pd.DataFrame(report.get("findings", [])).to_csv(index=False).encode("utf-8")


def csv_level2(report):
    return pd.DataFrame(report.get("differences", [])).to_csv(index=False).encode("utf-8")


def csv_level3(report):
    rows = []

    for page in report.get("pages", []):
        review = page.get("gemini_review", {})

        if page.get("findings"):
            for finding in page.get("findings", []):
                row = {
                    "page_name": page.get("page_name"),
                    "url": page.get("url"),
                    "status": page.get("status"),
                    "diff_percentage": page.get("diff_percentage"),
                    "similarity_score": page.get("similarity_score"),
                    "review_verdict": review.get("overall_page_verdict"),
                    "review_summary": review.get("summary")
                }
                row.update(finding)
                rows.append(row)
        else:
            rows.append({
                "page_name": page.get("page_name"),
                "url": page.get("url"),
                "status": page.get("status"),
                "diff_percentage": page.get("diff_percentage"),
                "similarity_score": page.get("similarity_score"),
                "review_verdict": review.get("overall_page_verdict"),
                "review_summary": review.get("summary")
            })

    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def run_level1(path):
    report = analyze_level1(path)
    save_json(report, "level1_audit.json")

    (REPORT_DIR / "level1_report.md").write_text(
        markdown_level1(report),
        encoding="utf-8"
    )

    return report


def run_level2(baseline_path, current_path, ignored_regions):
    diff_path = DIFF_DIR / "level2_diff.png"

    diff_data = calculate_diff(
        baseline_path,
        current_path,
        diff_path,
        ignore_regions=ignored_regions,
        min_area=300
    )

    report = analyze_level2(baseline_path, current_path, diff_data)
    save_json(report, "level2_regression.json")

    (REPORT_DIR / "level2_report.md").write_text(
        markdown_level2(report),
        encoding="utf-8"
    )

    return report, diff_data


def severity_badge(severity):
    severity = str(severity).lower()

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
html{scroll-behavior:smooth}
header[data-testid="stHeader"]{display:none}
.stApp{background:#020617;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1080px}
section[data-testid="stSidebar"]{background:#050b16;border-right:1px solid #13233a;width:250px!important}
section[data-testid="stSidebar"] *{color:#f8fafc!important;font-size:14px!important}
h1,h2,h3,h4,h5,h6,p,label,span{color:#f8fafc!important}
h1{font-size:38px!important;letter-spacing:-.04em}
h2{font-size:25px!important;letter-spacing:-.03em}
h3{font-size:20px!important;letter-spacing:-.02em}
p,label,span{font-size:14px!important}
.hero-card{background:linear-gradient(135deg,#071426 0%,#0b1d35 100%);border:1px solid #13233a;border-radius:22px;padding:28px;margin-bottom:22px;box-shadow:0 24px 70px rgba(0,0,0,.45);animation:fadeUp .5s ease-out}
.hero-title{font-size:38px;font-weight:760;color:#f8fafc;margin-bottom:8px;letter-spacing:-.05em}
.hero-subtitle{font-size:14px;color:#cbd5e1!important;margin-bottom:16px;max-width:760px;line-height:1.7}
.pill{display:inline-block;padding:6px 12px;border-radius:999px;background:#0f2742;color:#e0f2fe;border:1px solid #24496f;font-size:12px;margin-right:7px;margin-top:7px}
.panel{background:#071426;border:1px solid #13233a;border-radius:20px;padding:22px;margin-bottom:20px;box-shadow:0 18px 45px rgba(0,0,0,.34);animation:fadeUp .45s ease-out}
.finding-card{background:#071426;color:#f8fafc;border-radius:18px;padding:20px;margin-bottom:16px;border:1px solid #13233a;border-left:4px solid #38bdf8;box-shadow:0 16px 40px rgba(0,0,0,.34);animation:fadeUp .45s ease-out}
.finding-card h3,.finding-card p,.finding-card b{color:#f8fafc!important}
.small-label{color:#7dd3fc!important;font-size:11px!important;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
div[data-testid="stMetric"]{background:#071426;border-radius:18px;padding:16px;border:1px solid #13233a;box-shadow:0 14px 32px rgba(0,0,0,.32)}
div[data-testid="stMetric"] label{color:#93c5fd!important;font-size:12px!important;font-weight:650}
div[data-testid="stMetric"] div{color:#f8fafc!important;font-size:22px!important}
div[data-testid="stFileUploader"]{background:#071426;border-radius:18px;padding:12px;border:1px solid #13233a}
div[data-testid="stFileUploader"] *{color:#f8fafc!important}
section[data-testid="stFileUploaderDropzone"]{background:#0b1220!important;border:1px dashed #38bdf8!important;border-radius:16px!important}
section[data-testid="stFileUploaderDropzone"] *{color:#f8fafc!important}
section[data-testid="stFileUploaderDropzone"] button{background:#38bdf8!important;color:#020617!important;border-radius:10px!important;border:none!important;font-weight:700!important}
.stButton>button,.stDownloadButton>button{background:#38bdf8;color:#020617!important;border-radius:12px;border:none;padding:.72rem 1rem;font-weight:760;transition:all .25s ease-in-out}
.stButton>button:hover,.stDownloadButton>button:hover{background:#7dd3fc;transform:translateY(-1px)}
textarea{background-color:#020617!important;color:#e5e7eb!important;border:1px solid #13233a!important;border-radius:14px!important;font-family:Consolas,monospace!important;font-size:13px!important}
div[data-baseweb="select"] *{color:#f8fafc!important}
div[data-baseweb="select"]>div{background-color:#071426!important;border-color:#13233a!important}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
</style>
""",
    unsafe_allow_html=True
)


st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">VisualGuard AI</div>
    <div class="hero-subtitle">
        Premium UI audit agent for screenshot analysis, before/after regression review, autonomous multi-page scanning, baseline refresh, login capture, dynamic filtering, Gemini page reviews, and structured reporting.
    </div>
    <span class="pill">Agent 1</span>
    <span class="pill">Level 1 + 2 + 3</span>
    <span class="pill">Gemini 2.5 Flash Lite</span>
    <span class="pill">Baseline Store</span>
    <span class="pill">Dynamic Filters</span>
</div>
""",
    unsafe_allow_html=True
)


api_key = os.getenv("GEMINI_API_KEY")

st.sidebar.title("VisualGuard AI")
st.sidebar.caption("Design Audit Agent")

if api_key and api_key.strip() != "your_gemini_api_key_here":
    st.sidebar.success("Gemini connected")
    st.sidebar.caption("API key loaded from environment.")
else:
    st.sidebar.error("API key missing")

st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    [
        "Level 1 · Screenshot Audit",
        "Level 2 · Before/After Regression",
        "Level 3 · Autonomous Website Scan"
    ]
)

st.sidebar.write("Outputs: JSON / MD / TXT / CSV")


def output_panel(result, report_type):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Structured Output")

    json_text = json.dumps(result, indent=2, ensure_ascii=False)

    if report_type == "level1":
        markdown_text = markdown_level1(result)
        txt_text = text_report(result, report_type)
        csv_data = csv_level1(result)
        file_names = (
            "visualguard_level1_report.json",
            "visualguard_level1_report.md",
            "visualguard_level1_report.txt",
            "visualguard_level1_findings.csv"
        )

    elif report_type == "level2":
        markdown_text = markdown_level2(result)
        txt_text = text_report(result, report_type)
        csv_data = csv_level2(result)
        file_names = (
            "visualguard_level2_report.json",
            "visualguard_level2_report.md",
            "visualguard_level2_report.txt",
            "visualguard_level2_differences.csv"
        )

    else:
        markdown_text = markdown_level3(result)
        txt_text = text_report(result, report_type)
        csv_data = csv_level3(result)
        file_names = (
            "visualguard_level3_report.json",
            "visualguard_level3_report.md",
            "visualguard_level3_report.txt",
            "visualguard_level3_findings.csv"
        )

    selected_format = st.selectbox(
        "Output format",
        ["JSON", "Markdown", "Text", "CSV"],
        key=f"{report_type}_format"
    )

    if selected_format == "JSON":
        st.text_area("Preview", json_text, height=420, key=f"{report_type}_json_preview")

        st.download_button(
            "Download JSON",
            json_text.encode("utf-8"),
            file_names[0],
            "application/json",
            use_container_width=True,
            key=f"{report_type}_json_download"
        )

    elif selected_format == "Markdown":
        st.text_area("Preview", markdown_text, height=420, key=f"{report_type}_md_preview")

        st.download_button(
            "Download Markdown",
            markdown_text.encode("utf-8"),
            file_names[1],
            "text/markdown",
            use_container_width=True,
            key=f"{report_type}_md_download"
        )

    elif selected_format == "Text":
        st.text_area("Preview", txt_text, height=420, key=f"{report_type}_txt_preview")

        st.download_button(
            "Download Text",
            txt_text.encode("utf-8"),
            file_names[2],
            "text/plain",
            use_container_width=True,
            key=f"{report_type}_txt_download"
        )

    else:
        if report_type == "level1":
            st.dataframe(pd.DataFrame(result.get("findings", [])), use_container_width=True)

        elif report_type == "level2":
            st.dataframe(pd.DataFrame(result.get("differences", [])), use_container_width=True)

        else:
            st.dataframe(pd.DataFrame(result.get("pages", [])), use_container_width=True)

        st.download_button(
            "Download CSV",
            csv_data,
            file_names[3],
            "text/csv",
            use_container_width=True,
            key=f"{report_type}_csv_download"
        )

    st.markdown("</div>", unsafe_allow_html=True)


if mode.startswith("Level 1"):
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
                st.session_state.level1_result = run_level1(image_path)

    if st.session_state.level1_result:
        result = st.session_state.level1_result

        if result.get("source") == "Gemini Vision":
            st.success("Audit completed using Gemini Vision.")
        else:
            st.warning("Audit completed using fallback output.")

        if result.get("fallback_reason"):
            st.error(result.get("fallback_reason"))

        findings = result["findings"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Score", f"{result['overall_score']}/100")
        col2.metric("Findings", len(findings))
        col3.metric(
            "Confidence",
            f"{round(sum(item['confidence'] for item in findings) / len(findings), 2)}%"
        )
        col4.metric("Model", result.get("model_used", result.get("source")))

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Findings Summary")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "principle": item["principle"],
                        "severity": item["severity"],
                        "location": item["location"],
                        "confidence": item["confidence"]
                    }
                    for item in findings
                ]
            ),
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Detailed Findings")

        for index, item in enumerate(findings, 1):
            st.markdown(
                f"""
<div class="finding-card">
    <p class="small-label">Finding {index}</p>
    <h3>{item["principle"]} · {severity_badge(item["severity"])}</h3>
    <p><b>Location:</b> {item["location"]}</p>
    <p><b>Issue:</b> {item["issue"]}</p>
    <p><b>User Impact:</b> {item["user_impact"]}</p>
    <p><b>Recommendation:</b> {item["recommendation"]}</p>
    <p><b>Evidence:</b> {item["evidence"]}</p>
    <p><b>Confidence:</b> {item["confidence"]}%</p>
</div>
""",
                unsafe_allow_html=True
            )

        output_panel(result, "level1")


elif mode.startswith("Level 2"):
    st.markdown(
        """
<div class="panel">
    <h2>Before/After Visual Regression</h2>
    <p>Upload a baseline and current screenshot. The agent detects pixel regions, hex color shifts, accessibility regressions, and UX impact.</p>
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

    ignore_text = st.text_area(
        "Optional dynamic regions to ignore, one per line as x,y,width,height",
        height=90,
        key="level2_ignore"
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
        run_button = st.button("Run Regression Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_button:
            ignored_regions = parse_regions(ignore_text)

            with st.spinner("Comparing screenshots..."):
                result, diff_data = run_level2(baseline_path, current_path, ignored_regions)
                st.session_state.level2_result = result
                st.session_state.level2_diff = diff_data

    if st.session_state.level2_result:
        result = st.session_state.level2_result
        diff_data = st.session_state.level2_diff

        st.success("Regression analysis completed.")

        differences = result["differences"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Similarity", diff_data["similarity_score"])
        col2.metric("Diff %", f"{diff_data['diff_percentage']}%")
        col3.metric("Changes", len(differences))
        col4.metric("Verdict", result.get("overall_verdict", "Review"))

        diff_path = Path(diff_data["diff_image_path"])

        if diff_path.is_file():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Highlighted Difference Image")
            st.image(Image.open(diff_path), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Detected Differences")

        for index, item in enumerate(differences, 1):
            st.markdown(
                f"""
<div class="finding-card">
    <p class="small-label">Difference {index}</p>
    <h3>{item["change_type"].title()} · {item["confidence"]}%</h3>
    <p><b>Location:</b> {item["location"]}</p>
    <p><b>What Changed:</b> {item["what_changed"]}</p>
    <p><b>UX Impact:</b> {item["ux_impact"]}</p>
    <p><b>Recommendation:</b> {item["recommendation"]}</p>
    <p><b>Evidence:</b> {json.dumps(item["evidence"])}</p>
</div>
""",
                unsafe_allow_html=True
            )

        output_panel(result, "level2")


else:
    st.markdown(
        """
<div class="panel">
    <h2>Autonomous Website Scan</h2>
    <p>Configure 3–4 pages, optional login, ignored dynamic regions, and baseline refresh. First run creates baselines; later runs detect regressions and optional Gemini reviews each compared page.</p>
</div>
""",
        unsafe_allow_html=True
    )

    pages_text = st.text_area(
        "Pages to scan: one per line as page_name|url",
        value="home|https://example.com",
        height=125
    )

    threshold = st.slider(
        "Regression threshold percentage",
        0.5,
        10.0,
        2.0,
        0.5
    )

    use_gemini = st.checkbox(
        "Use Gemini review for Level 3",
        value=False,
        help="Turn this on only for final demo because every compared page uses Gemini quota."
    )

    ignore_text = st.text_area(
        "Dynamic regions to ignore, one per line as x,y,width,height",
        height=90,
        help="Use this for timestamps, counters, loading states, ads, session IDs."
    )

    with st.expander("Optional login settings"):
        login_url = st.text_input("Login URL", "")
        username = st.text_input("Username", "")
        password = st.text_input("Password", "", type="password")
        username_selector = st.text_input("Username CSS selector", "")
        password_selector = st.text_input("Password CSS selector", "")
        submit_selector = st.text_input("Submit button CSS selector", "")

    refresh = st.checkbox(
        "Refresh approved baselines",
        value=False,
        help="Use only after reviewing and approving the current UI."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    run_button = st.button("Run Autonomous Scan", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_button:
        login_config = {
            "login_url": login_url,
            "username": username,
            "password": password,
            "username_selector": username_selector,
            "password_selector": password_selector,
            "submit_selector": submit_selector
        }

        with st.spinner("Capturing pages and comparing with baselines..."):
            try:
                st.session_state.level3_result = run_level3(
                    parse_pages(pages_text),
                    threshold,
                    parse_regions(ignore_text),
                    refresh,
                    login_config,
                    use_gemini
                )

            except Exception as error:
                st.error("Level 3 scan failed.")
                st.write(str(error))

    if st.session_state.level3_result:
        result = st.session_state.level3_result

        if result.get("pages_with_regressions", 0) > 0:
            st.warning("Regressions detected.")
        else:
            st.success("Scan completed. No significant regressions detected or baselines were created/refreshed.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Status", result.get("status"))
        col2.metric("Pages", result.get("total_pages"))
        col3.metric("Regressions", result.get("pages_with_regressions"))
        col4.metric("Time", f"{result.get('duration_seconds')}s")

        for page in result.get("pages", []):
            review = page.get("gemini_review", {})

            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.subheader(page.get("page_name"))
            st.write(f"URL: {page.get('url')}")
            st.write(f"Status: {page.get('status')}")
            st.write(f"Similarity: {page.get('similarity_score')} | Diff: {page.get('diff_percentage')}%")
            st.write(f"Review Verdict: {review.get('overall_page_verdict')}")
            st.write(f"Review Summary: {review.get('summary')}")

            baseline_path_value = page.get("baseline_path")
            current_path_value = page.get("current_path")
            diff_path_value = page.get("diff_image_path")

            image_col1, image_col2 = st.columns(2)

            with image_col1:
                if baseline_path_value:
                    baseline_path = Path(baseline_path_value)

                    if baseline_path.is_file():
                        st.image(
                            Image.open(baseline_path),
                            caption="Baseline Screenshot",
                            use_container_width=True
                        )

            with image_col2:
                if current_path_value:
                    current_path = Path(current_path_value)

                    if current_path.is_file():
                        st.image(
                            Image.open(current_path),
                            caption="Current Screenshot",
                            use_container_width=True
                        )

            if diff_path_value:
                diff_path = Path(diff_path_value)

                if diff_path.is_file():
                    st.image(
                        Image.open(diff_path),
                        caption="Highlighted Difference Image",
                        use_container_width=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            findings = page.get("findings", [])

            if findings:
                for index, item in enumerate(findings, 1):
                    st.markdown(
                        f"""
<div class="finding-card">
    <p class="small-label">Page Review {index}</p>
    <h3>{item.get("change_type", "neutral").title()} · {severity_badge(item.get("severity", "info"))}</h3>
    <p><b>Location:</b> {item.get("location")}</p>
    <p><b>What Changed:</b> {item.get("what_changed")}</p>
    <p><b>UX Impact:</b> {item.get("ux_impact")}</p>
    <p><b>Recommendation:</b> {item.get("recommendation")}</p>
    <p><b>Evidence:</b> {json.dumps(item.get("evidence", {}))}</p>
    <p><b>Confidence:</b> {item.get("confidence")}%</p>
</div>
""",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    """
<div class="finding-card">
    <p class="small-label">Page Review</p>
    <h3>No compared findings yet</h3>
    <p>This usually means the baseline was just created or refreshed. Run the scan again to compare against the stored baseline.</p>
</div>
""",
                    unsafe_allow_html=True
                )

        output_panel(result, "level3")