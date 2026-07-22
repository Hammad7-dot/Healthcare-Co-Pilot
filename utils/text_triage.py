"""
TEXT modality (3rd of the required 3+ modalities): free-text patient symptom notes.

Two layers, matching the hackathon rules:
1. A lightweight, transparent keyword/rule-based symptom extractor. This is NOT
   the primary predictive engine -- it flags symptom keywords and urgency terms
   so the co-pilot has structured signal from unstructured text.
2. An OPTIONAL LLM call (Anthropic API) used strictly for reasoning /
   summarization / explanation / report narrative generation -- never as the
   predictive engine, per hackathon rules ("LLMs may be used for reasoning,
   summarization, explanations, or report generation but not as the primary
   predictive engine").

If no ANTHROPIC_API_KEY is set, the LLM step is skipped and a rule-based
summary is used instead, so the app still runs fully offline.
"""

import os
import re

SYMPTOM_KEYWORDS = {
    "chest_pain": ["chest pain", "chest tightness", "chest pressure"],
    "shortness_of_breath": ["shortness of breath", "breathless", "can't breathe", "difficulty breathing"],
    "cough": ["cough", "coughing", "phlegm", "sputum"],
    "fever": ["fever", "chills", "high temperature"],
    "fatigue": ["fatigue", "tired", "exhausted", "low energy"],
    "dizziness": ["dizzy", "dizziness", "lightheaded"],
    "palpitations": ["palpitations", "racing heart", "irregular heartbeat"],
    "swelling": ["swelling", "edema", "swollen legs", "swollen ankles"],
}

URGENT_TERMS = ["severe chest pain", "can't breathe", "crushing pain",
                "fainted", "blue lips", "collapsed"]


def extract_symptoms(note_text: str):
    text = note_text.lower()
    found = {}
    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found[symptom] = True
                break
    urgent_flags = [term for term in URGENT_TERMS if term in text]
    return {"symptoms": list(found.keys()), "urgent_flags": urgent_flags}


def rule_based_summary(extraction: dict) -> str:
    symptoms = extraction["symptoms"]
    urgent = extraction["urgent_flags"]
    if not symptoms:
        return "No specific symptom keywords detected in the free-text note."
    text = "Detected symptom signals: " + ", ".join(s.replace("_", " ") for s in symptoms) + "."
    if urgent:
        text += " URGENT LANGUAGE DETECTED -- recommend immediate clinical review."
    return text


def llm_summary(note_text: str, extraction: dict, model_prediction_context: str = "") -> str:
    """
    Optional: calls Anthropic API to turn structured findings into a clinician-
    readable narrative. Falls back to rule_based_summary if no API key is present
    or the call fails, so this is never a hard dependency.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_based_summary(extraction)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are assisting a clinician-facing decision support tool. "
            "Given this patient free-text note and extracted symptom signals, "
            "write a 2-3 sentence neutral clinical summary. Do not diagnose; "
            "only summarize signals and note where model predictions suggest "
            "further review.\n\n"
            f"Patient note: {note_text}\n"
            f"Extracted symptoms: {extraction['symptoms']}\n"
            f"Urgent flags: {extraction['urgent_flags']}\n"
            f"Model context: {model_prediction_context}"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
    except Exception:
        return rule_based_summary(extraction)
