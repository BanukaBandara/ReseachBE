from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Advice:
    title: str
    confidence_note: str
    what_to_check: list[str]
    recovery_steps: list[str]
    when_to_escalate: list[str]


def generate_farmer_advice(
    *,
    health_label: str,
    health_confidence: float,
    growth_stage_month: int | None = None,
    stunted_flag: bool | None = None,
) -> Advice:
    """Generate safe, general farmer guidance based on prediction.

    This is intentionally conservative (no exact fertilizer rates) because
    correct actions depend on soil type, rainfall/irrigation, cultivar, and severity.
    """

    # Confidence note
    if health_confidence >= 0.75:
        conf_note = "High confidence prediction based on the photo."
    elif health_confidence >= 0.55:
        conf_note = "Medium confidence prediction. Consider confirming with a quick field check."
    else:
        conf_note = (
            "Low confidence prediction (image conditions may be hard). Take another photo in good light "
            "and consider expert confirmation."
        )

    common_checks = [
        "Take 2–3 photos from different angles (whole plant + close-up leaf).",
        "Check if symptoms appear on older leaves first or newer leaves first.",
        "Check soil moisture near the root zone (not just surface).",
    ]

    common_escalate = [
        "If symptoms spread quickly within 3–7 days.",
        "If the plant wilts severely or the crown/roots look unhealthy.",
        "If you suspect pests/disease not covered by the model.",
        "Consult a local agronomist if unsure—local conditions matter.",
    ]

    title = f"Suggested next steps for: {health_label.replace('_', ' ')}"

    what_to_check: list[str] = list(common_checks)
    recovery_steps: list[str] = []
    when_to_escalate: list[str] = list(common_escalate)

    if health_label == "nitrogen_deficiency":
        what_to_check.extend(
            [
                "Look for uniform yellowing/paling, often starting on older leaves.",
                "Compare several plants in the field—nitrogen deficiency often shows across an area.",
            ]
        )
        recovery_steps.extend(
            [
                "Apply a nitrogen-containing fertilizer in small split applications (follow local label/extension guidance).",
                "Avoid applying just before heavy rain to reduce losses.",
                "Maintain steady irrigation (avoid both drought and waterlogging).",
                "Re-check after 7–14 days with a new photo to confirm improvement.",
            ]
        )

    elif health_label == "water_stress":
        what_to_check.extend(
            [
                "Check for leaf curl/rolling or dull/grey-green appearance during hot hours.",
                "Inspect irrigation lines/drippers for blockages and uneven coverage.",
            ]
        )
        recovery_steps.extend(
            [
                "Adjust irrigation to deeper, less frequent watering (based on soil type) rather than many tiny waterings.",
                "Water early morning/late afternoon to reduce evaporation.",
                "Add mulch around plants (not covering the crown) to reduce moisture loss.",
                "If soil stays wet for long periods, reduce watering and improve drainage.",
            ]
        )

    elif health_label == "healthy":
        what_to_check.extend(
            [
                "Keep monitoring weekly/biweekly—early issues are easier to fix.",
                "Maintain consistent nutrition and irrigation schedules.",
            ]
        )
        recovery_steps.extend(
            [
                "Continue normal best practices for your farm.",
                "If you notice new yellowing or curl, retake a photo and run prediction again.",
            ]
        )

    else:
        # Unknown label (future-proof)
        recovery_steps.extend(
            [
                "Retake the photo with better lighting and a clear view of the leaves.",
                "If the problem persists, consult a local agronomist for a proper diagnosis.",
            ]
        )

    if stunted_flag is True:
        recovery_steps.insert(
            0,
            "The system also flags possible stunted growth. Check water, nutrition, and root health together—stunting often has multiple causes.",
        )
        what_to_check.extend(
            [
                "Compare plant size to nearby plants of the same age.",
                "Check for compacted soil, poor drainage, or root restriction.",
            ]
        )

    if growth_stage_month is not None:
        what_to_check.append(f"Predicted growth stage is month {growth_stage_month}. Use this as a guide, not an absolute truth.")

    return Advice(
        title=title,
        confidence_note=conf_note,
        what_to_check=what_to_check,
        recovery_steps=recovery_steps,
        when_to_escalate=when_to_escalate,
    )


def advice_to_dict(advice: Advice) -> dict[str, Any]:
    return {
        "title": advice.title,
        "confidence_note": advice.confidence_note,
        "what_to_check": advice.what_to_check,
        "recovery_steps": advice.recovery_steps,
        "when_to_escalate": advice.when_to_escalate,
    }
