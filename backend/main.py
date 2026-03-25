"""
ScamShield Hybrid Backend
=========================
Scoring logic: rule-based NLP (regex) + TF-IDF + Logistic Regression
Claude API: ONLY used for natural-language explanations (not scoring)

Run:
    pip install fastapi uvicorn scikit-learn numpy
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import re, math, json
from collections import Counter
from datetime import datetime

app = FastAPI(title="ScamShield Hybrid API", version="2.0")

# Allow requests from the HTML frontend (any origin for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# SHARED NLP UTILITIES
# ──────────────────────────────────────────────

# ── Rule-Based Pattern Library ──
PAYMENT_PATTERNS = [
    r'\b(pay|payment|fee|deposit|charge|cost|investment|registration)\b',
    r'₹\s*\d+', r'\$\s*\d+', r'\b\d+\s*(rs|inr|usd|dollars?)\b',
    r'\b(send money|transfer|paytm|gpay|upi|phonepay|wire transfer)\b',
    r'\b(refundable|non-refundable|security deposit|advance)\b',
]

URGENCY_PATTERNS = [
    r'\b(urgent|urgently|immediate|immediately|asap|hurry|rush)\b',
    r'\b(limited (seats?|slots?|positions?|time|offer))\b',
    r'\b(only today|expires? (today|soon|now)|last chance|final call)\b',
    r'\b(apply (now|immediately|today|asap))\b',
    r'\b(don\'?t miss|act (now|fast|quickly)|time (is )?running out)\b',
    r'\b(confirm (your )?seat|grab (this|your) (slot|opportunity))\b',
]

NO_INTERVIEW_PATTERNS = [
    r'\b(no interview|without interview|skip (the )?interview)\b',
    r'\b(directly (selected|hired|confirmed))\b',
    r'\b(selected without|confirmed without|hired without)\b',
    r'\b(instant (selection|hiring|offer))\b',
]

UNREALISTIC_SALARY_PATTERNS = [
    r'(earn|salary|income|pay(ing)?|get paid)[\s\S]{0,40}(\d{4,})',
    r'(₹|rs\.?|inr)\s*\d{2,}[,\s]*\d{3}',
    r'\b\d+k?\s*(per\s*)?(day|week|month|hr|hour)\b',
    r'\b(lakhs?|crores?)\b',
    r'\$\s*\d{4,}',
]

SUSPICIOUS_CONTACT_PATTERNS = [
    r'\b(whatsapp|telegram|signal)\b.*\b\d{10}\b',
    r'\b\d{10,}\b',
    r'\b[a-z0-9._%+-]+@(gmail|yahoo|hotmail|outlook|rediff|ymail)\.',
    r'\b(call|contact|message|reach)\s+(us|me|hr)\s+on\s+(whatsapp|telegram)\b',
]

PRESSURE_LANGUAGE = [
    r'\b(congratulations?|you (are|have been) (selected|chosen|shortlisted))\b',
    r'\b(once in a lifetime|golden (chance|opportunity))\b',
    r'\b(100%|guaranteed|assured)\s+(job|income|salary|earning|placement)\b',
    r'\b(no (experience|qualification|skill)s? (needed|required))\b',
    r'\b(work from home|wfh|remote)\b.*\b(easy|simple|anyone)\b',
    r'\b(part[- ]time|part time)\b.*₹\s*\d+',
]

LEGIT_SIGNALS = [
    r'\b(linkedin\.com|glassdoor|naukri|indeed|monster)\b',
    r'\b(schedule|book|calendar).*\b(interview|call|meeting)\b',
    r'\b(hr|recruiter|talent acquisition)\s+at\s+\w+\b',
    r'\b(offer letter|employment contract|background check|onboarding)\b',
    r'\b(probation|notice period|annual (leave|increment))\b',
    r'\b(@[a-z0-9-]+\.(com|in|org|co\.in))\b',  # company email domain
]

def count_pattern_hits(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))

def extract_matched_phrases(text: str, patterns: list) -> list:
    text_lower = text.lower()
    found = []
    for p in patterns:
        m = re.search(p, text_lower)
        if m:
            found.append(m.group(0).strip())
    return list(set(found))





# Build and train the mini ML model at startup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset
df = pd.read_csv("fake_job_postings.csv")
df = df[['description', 'fraudulent']].dropna()

texts = df['description'].apply(clean_text).tolist()[:15000]
labels = df['fraudulent'].tolist()[:15000]

# Train model
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=7000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=300, class_weight='balanced')
model.fit(X, labels)

def ml_score(text: str) -> float:
    vec = vectorizer.transform([clean_text(text)])
    return model.predict_proba(vec)[0][1]


def rule_score(text: str) -> dict:
    """Returns component scores (0–100) from regex rule engine."""
    payment_hits    = count_pattern_hits(text, PAYMENT_PATTERNS)
    urgency_hits    = count_pattern_hits(text, URGENCY_PATTERNS)
    no_int_hits     = count_pattern_hits(text, NO_INTERVIEW_PATTERNS)
    salary_hits     = count_pattern_hits(text, UNREALISTIC_SALARY_PATTERNS)
    contact_hits    = count_pattern_hits(text, SUSPICIOUS_CONTACT_PATTERNS)
    pressure_hits   = count_pattern_hits(text, PRESSURE_LANGUAGE)
    legit_hits      = count_pattern_hits(text, LEGIT_SIGNALS)

    payment_score   = min(100, payment_hits * 40)
    urgency_score   = min(100, urgency_hits * 20)
    no_int_score    = min(100, no_int_hits  * 35)
    salary_score    = min(100, salary_hits  * 25)
    contact_score   = min(100, contact_hits * 30)
    pressure_score  = min(100, pressure_hits * 20)
    legit_reduction = min(60, legit_hits * 20)

    raw = (payment_score * 0.35 +
           urgency_score * 0.15 +
           no_int_score  * 0.20 +
           salary_score  * 0.10 +
           contact_score * 0.10 +
           pressure_score * 0.10)

    final = max(0, raw - legit_reduction)

    return {
        "payment_risk":   round(payment_score),
        "urgency_risk":   round(urgency_score),
        "no_interview_risk": round(no_int_score),
        "salary_risk":    round(salary_score),
        "contact_risk":   round(contact_score),
        "pressure_risk":  round(pressure_score),
        "legit_reduction": round(legit_reduction),
        "rule_score":     round(min(100, final)),
    }


def hybrid_score(text: str) -> tuple:
    """
    Combine rule-based (60%) + ML (40%) → final 0-100 score.
    Returns (final_score, component_breakdown).
    """
    rules  = rule_score(text)
    ml_raw = ml_score(text) * 100  # 0–100

    # Weighted fusion
    combined = rules["rule_score"] * 0.75 + ml_raw * 0.25
    final = round(min(100, max(0, combined)))

    level = "HIGH" if final >= 60 else "MEDIUM" if final >= 25 else "LOW"
    verdict = "Likely Scam" if final >= 60 else "Suspicious" if final >= 25 else "Safe"

    return final, level, verdict, rules, round(ml_raw)


def get_suspicious_phrases(text: str) -> list:
    found = []
    for patterns in [PAYMENT_PATTERNS, URGENCY_PATTERNS, NO_INTERVIEW_PATTERNS, PRESSURE_LANGUAGE]:
        found.extend(extract_matched_phrases(text, patterns))
    return list(set(found))[:8]


def get_red_flags(rules: dict, text: str) -> list:
    flags = []
    if rules["payment_risk"] >= 40:
        flags.append("Payment or fee request detected — legitimate employers never charge candidates")
    if rules["urgency_risk"] >= 40:
        flags.append("High-pressure urgency language (limited seats, apply now, expires today)")
    if rules["no_interview_risk"] >= 35:
        flags.append("Claims of selection without any interview process — a major red flag")
    if rules["salary_risk"] >= 25:
        flags.append("Unrealistic salary claims — figures appear significantly above market rates")
    if rules["contact_risk"] >= 30:
        flags.append("Contact via personal WhatsApp/Telegram or free email (Gmail/Yahoo) rather than company domain")
    if rules["pressure_risk"] >= 40:
        flags.append("Manipulative pressure language designed to bypass critical thinking")
    if not flags:
        if re.search(r'\b(gmail|yahoo|hotmail)\b', text.lower()):
            flags.append("Personal email used instead of company domain")
        else:
            flags.append("No major structural red flags found — review content carefully")
    return flags[:5]


# ──────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class DomainRequest(BaseModel):
    domain: str
    age_days: int

class TimelineRequest(BaseModel):
    messages: List[str]

class CrowdRequest(BaseModel):
    text: str
    similar_reports: int = 0

class HeatmapRequest(BaseModel):
    reports: List[str]  # ["City — description", ...]


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "TF-IDF + LogReg + Regex NLP"}


# ── /analyze/message  (ScamRadar single message) ──
@app.post("/analyze/message")
def analyze_message(req: TextRequest):
    score, level, verdict, rules, ml_raw = hybrid_score(req.text)
    phrases  = get_suspicious_phrases(req.text)
    flags    = get_red_flags(rules, req.text)
    return {
        "score":               score,
        "risk_level":          level,
        "verdict":             verdict,
        "suspicious_phrases":  phrases,
        "reasons":             flags,
        "components":          rules,
        "ml_score":            ml_raw,
        # summary is left empty — frontend will ask Claude for explanation
        "summary":             ""
    }


# ── /analyze/domain ──
@app.post("/analyze/domain")
def analyze_domain(req: DomainRequest):
    age = req.age_days
    if age < 90:
        level, score = "HIGH",   min(95, 90 + max(0, (90  - age)))
    elif age < 365:
        level, score = "MEDIUM", round(30 + (365 - age) / 365 * 40)
    else:
        level, score = "LOW",    max(5, round(30 - (age - 365) / 365 * 20))

    domain_flags = []
    d = req.domain.lower()
    if re.search(r'(job|hire|employ|career|work|earn|salary)', d):
        score = min(100, score + 10)
        domain_flags.append("Domain name contains job-related keywords — common in scam sites")
    if re.search(r'(-|\d{4,})', d):
        score = min(100, score + 5)
        domain_flags.append("Hyphenated or numeric domain pattern detected")
    if re.search(r'\.(xyz|tk|ml|ga|cf|gq|top|click|online)$', d):
        score = min(100, score + 15)
        domain_flags.append("Uses a free or low-trust TLD (e.g., .xyz, .tk)")

    return {
        "domain":      req.domain,
        "age_days":    age,
        "risk_level":  level,
        "score":       round(score),
        "domain_flags": domain_flags,
        # explanation left empty for Claude
        "reason":      "",
        "additional_tip": ""
    }


# ── /analyze/url ──
@app.post("/analyze/url")
def analyze_url(req: TextRequest):
    # Split input — first line treated as URL, rest as content
    lines = req.text.strip().split("\n")
    url = lines[0].strip() if lines else ""
    content = "\n".join(lines[1:]).strip() if len(lines) > 1 else req.text

    url_score = 0
    url_flags = []

    if url:
        if re.search(r'\b(earn|hire|job|work|money|salary|cash)\b', url.lower()):
            url_score += 25
            url_flags.append("URL contains income/job keywords")
        if re.search(r'(urgent|now|fast|free|quick)', url.lower()):
            url_score += 20
            url_flags.append("URL contains urgency keywords")
        if re.search(r'-{2,}|\d{4,}', url.lower()):
            url_score += 10
            url_flags.append("Suspicious URL structure (multiple hyphens or numbers)")
        if not re.search(r'https://', url.lower()):
            url_score += 10
            url_flags.append("No HTTPS — potentially insecure connection")

    content_score, level, verdict, rules, ml_raw = hybrid_score(content or url)
    final = round(min(100, content_score * 0.7 + url_score * 0.3))
    level = "HIGH" if final >= 65 else "MEDIUM" if final >= 35 else "LOW"
    verdict = "Likely Scam" if final >= 65 else "Suspicious" if final >= 35 else "Safe"

    reasons = url_flags + get_red_flags(rules, content)

    return {
        "scam_probability": final,
        "risk_level":       level,
        "verdict":          verdict,
        "reasons":          reasons[:4],
        "url_score":        url_score,
        "content_score":    content_score,
        "ml_score":         ml_raw,
        "components":       rules,
    }


# ── /analyze/timeline ──
@app.post("/analyze/timeline")
def analyze_timeline(req: TimelineRequest):
    timeline = []
    for i, msg in enumerate(req.messages):
        score, level, verdict, rules, ml_raw = hybrid_score(msg)
        # Classify escalation tag
        if rules["payment_risk"] >= 40:
            tag = "payment_demand"
        elif rules["urgency_risk"] >= 40 or rules["pressure_risk"] >= 40:
            tag = "pressure"
        elif score >= 35:
            tag = "suspicious"
        else:
            tag = "normal"
        timeline.append({
            "message": msg[:70] + ("..." if len(msg) > 70 else ""),
            "risk":    score,
            "tag":     tag,
            "index":   i + 1,
        })

    scores = [t["risk"] for t in timeline]
    first, last = scores[0] if scores else 0, scores[-1] if scores else 0
    overall = round(max(scores) * 0.6 + (sum(scores) / max(len(scores), 1)) * 0.4)
    overall_level = "HIGH" if overall >= 65 else "MEDIUM" if overall >= 35 else "LOW"

    return {
        "timeline":     timeline,
        "overall_risk": overall_level,
        "overall_score": overall,
        "first_score":  first,
        "last_score":   last,
        # summary left for Claude
        "summary":      ""
    }


# ── /analyze/gauge ──
@app.post("/analyze/gauge")
def analyze_gauge(req: TextRequest):
    score, level, verdict, rules, ml_raw = hybrid_score(req.text)
    label = "High Risk" if level == "HIGH" else "Medium Risk" if level == "MEDIUM" else "Low Risk"
    key_signals = get_suspicious_phrases(req.text)

    # Confidence: based on how decisive the score is (far from 50 = more confident)
    confidence = round(min(99, 50 + abs(score - 50) * 0.9))

    return {
        "scam_score":  score,
        "label":       label,
        "risk_level":  level,
        "confidence":  confidence,
        "key_signal":  key_signals[0] if key_signals else "pattern analysis complete",
        "ml_score":    ml_raw,
        "components":  rules,
    }


# ── /analyze/crowd ──
@app.post("/analyze/crowd")
def analyze_crowd(req: CrowdRequest):
    score, level, verdict, rules, ml_raw = hybrid_score(req.text)
    phrases = get_suspicious_phrases(req.text)

    # Known scam classification
    if score >= 70 or req.similar_reports >= 20:
        is_known = "Yes"
    elif score >= 40 or req.similar_reports >= 5:
        is_known = "Likely"
    else:
        is_known = "No"

    # Pattern match
    pattern_match = "Yes" if len(phrases) >= 3 else "Partial" if len(phrases) >= 1 else "No"

    # Scam type inference
    scam_type = "Unknown"
    if rules["payment_risk"] >= 40:
        scam_type = "Advance Fee Fraud"
    elif rules["no_interview_risk"] >= 35:
        scam_type = "Fake Job Offer"
    elif rules["contact_risk"] >= 30:
        scam_type = "Social Engineering / Phishing"
    elif rules["salary_risk"] >= 25:
        scam_type = "Work-From-Home Scam"
    elif rules["urgency_risk"] >= 40:
        scam_type = "High-Pressure Fraud"

    return {
        "is_known_scam":   is_known,
        "scam_type":       scam_type,
        "risk_level":      level,
        "score":           score,
        "similar_reports": req.similar_reports,
        "pattern_match":   pattern_match,
        "phrases":         phrases,
        "components":      rules,
        # explanation left for Claude
        "explanation":     ""
    }


# ── /analyze/heatmap ──
@app.post("/analyze/heatmap")
def analyze_heatmap(req: HeatmapRequest):
    city_counts = Counter()
    city_types = {}

    for report in req.reports:
        # Try to parse "City — description" or "City: description"
        parts = re.split(r'[—\-–:]', report, maxsplit=1)
        city = parts[0].strip().title() if parts else "Unknown"
        desc = parts[1].strip() if len(parts) > 1 else report

        city_counts[city] += 1
        if city not in city_types:
            city_types[city] = []

        # Tag type
        if re.search(r'\b(job|hire|employ|recruit)\b', desc, re.I):
            city_types[city].append("Job Fraud")
        elif re.search(r'\b(phishing|email|password|login)\b', desc, re.I):
            city_types[city].append("Phishing")
        elif re.search(r'\b(investment|crypto|bitcoin|trading)\b', desc, re.I):
            city_types[city].append("Investment Scam")
        elif re.search(r'\b(lottery|prize|won|winner)\b', desc, re.I):
            city_types[city].append("Lottery Scam")
        elif re.search(r'\b(upi|payment|transfer|bank)\b', desc, re.I):
            city_types[city].append("Payment Fraud")
        else:
            city_types[city].append("General Fraud")

    total = sum(city_counts.values())
    top5 = city_counts.most_common(5)

    dominant = Counter()
    for types in city_types.values():
        dominant.update(types)

    cities_out = []
    for city, count in top5:
        types = list(set(city_types.get(city, ["General Fraud"])))
        cities_out.append({
            "city":    city,
            "count":   count,
            "types":   types[:2],
            "pct":     round(count / max(total, 1) * 100),
            "insight": ""  # left for Claude
        })

    return {
        "top_cities":          cities_out,
        "total_reports":       total,
        "dominant_scam_type":  dominant.most_common(1)[0][0] if dominant else "Unknown",
        "city_distribution":   dict(city_counts.most_common()),
    }


# ── /analyze/breakdown ──
@app.post("/analyze/breakdown")
def analyze_breakdown(req: TextRequest):
    score, level, verdict, rules, ml_raw = hybrid_score(req.text)

    # Map rule components to breakdown factors
    salary_risk  = rules["salary_risk"]
    tone_risk    = round((rules["urgency_risk"] + rules["pressure_risk"]) / 2)
    domain_risk  = rules["contact_risk"]
    payment_risk = rules["payment_risk"]
    overall      = score

    return {
        "Salary Risk":        salary_risk,
        "Message Tone Risk":  tone_risk,
        "Domain Risk":        domain_risk,
        "Payment Risk":       payment_risk,
        "overall":            overall,
        "risk_level":         level,
        "ml_score":           ml_raw,
        "components":         rules,
        # summary left for Claude
        "summary":            ""
    }