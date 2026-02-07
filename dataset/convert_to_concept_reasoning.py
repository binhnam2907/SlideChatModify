"""
Convert WSI slide data to Concept-Based Reasoning format for Stage 2 training.

Training Philosophy:
  - The ONLY ground-truth label is "Final diagnosis" (KIRC / KIRP / KICH / N/A).
  - All other text (evidence analysis, presence scores, explanation) is
    auto-generated to teach the model visual reasoning during training.
    These sections are NOT used for scoring.
  - Accuracy is computed ONLY by extracting "Final diagnosis: <LABEL>"
    from the model output and comparing it to the ground truth.

Input:
  A CSV label file with two columns: slide_id, label
  Example:
      slide_id,label
      TCGA-A7-A0CJ-01Z-00-DX2,KIRC
      TCGA-AR-A255-01Z-00-DX1,KIRP

Usage:
    python convert_to_concept_reasoning.py \
        --label_map  label_mapping.csv \
        --image_dir  path/to/WSI_feat/ \
        --output     slidechat_train_concept_stage2.json
"""

import argparse
import json
import re
import os
import csv
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────

VALID_LABELS = {"KIRC", "KIRP", "KICH", "N/A"}

# The global instruction embedded into EVERY human prompt (identical for all cases).
# Contains comprehensive histological descriptions of ALL cancer types so the
# model always has the full reference context regardless of the specific slide.
GLOBAL_INSTRUCTION = (
    "Analyze the provided pathology slide. You must evaluate the presence of "
    "the following kidney cancer subtypes based on their histological features:\n"
    "\n"
    "KIRC (Kidney Renal Clear Cell Carcinoma):\n"
    "  - Clear cytoplasm due to glycogen and lipid accumulation\n"
    "  - Prominent delicate thin-walled vasculature (chicken-wire pattern)\n"
    "  - Alveolar or acinar architecture\n"
    "  - Cells arranged in nests surrounded by a rich vascular network\n"
    "  - Nuclei typically round with visible nucleoli (Fuhrman grading)\n"
    "\n"
    "KIRP (Kidney Renal Papillary Cell Carcinoma):\n"
    "  - Papillary architecture with fibrovascular cores\n"
    "  - Foamy macrophages within the papillary stalks\n"
    "  - Psammoma bodies (calcified concentric lamellae)\n"
    "  - Tubulo-papillary growth pattern\n"
    "  - Basophilic or eosinophilic cytoplasm depending on type (I vs II)\n"
    "\n"
    "KICH (Kidney Chromophobe):\n"
    "  - Pale eosinophilic or reticular cytoplasm\n"
    "  - Prominent perinuclear halos (clearing around nucleus)\n"
    "  - Plant-like cells with well-defined cell borders\n"
    "  - Raisinoid or wrinkled nuclei with irregular nuclear membranes\n"
    "  - Solid sheet or alveolar growth pattern\n"
    "\n"
    "N/A (No Cancer):\n"
    "  - Normal renal parenchyma without malignant features\n"
    "  - Preserved tubular and glomerular architecture\n"
    "  - No evidence of invasive carcinoma\n"
    "\n"
    "Follow the required diagnostic template in your response."
)

# ─── Auto-generated evidence text per label ───
# These are ONLY for training the model to produce visual reasoning.
# They are NOT used for scoring. Only "Final diagnosis" is scored.

_POSITIVE_EVIDENCE = {
    "KIRC": "Clear cell nests with delicate vascular network observed.",
    "KIRP": "Papillary architecture with fibrovascular cores observed.",
    "KICH": "Pale eosinophilic cells with prominent perinuclear halos observed.",
}

_NEGATIVE_EVIDENCE = {
    "KIRC": "No clear cell morphology or prominent vasculature identified.",
    "KIRP": "No papillary formations identified.",
    "KICH": "No perinuclear halos or pale eosinophilic cells present.",
}

_EXPLANATIONS = {
    "KIRC": "The presence of clear cytoplasm and prominent vasculature supports the diagnosis of KIRC.",
    "KIRP": "The presence of papillary structures supports the diagnosis of KIRP.",
    "KICH": "The presence of pale eosinophilic cells and perinuclear halos supports the diagnosis of KICH.",
    "N/A":  "No histological features indicative of renal cell carcinoma subtypes were identified.",
}


# ─────────────────────────────────────────────────────────────────────
# 2. Prompt & Output Builders
# ─────────────────────────────────────────────────────────────────────

def build_human_prompt() -> str:
    """
    Build the standardised human (instruction) prompt.
    The <image> token is appended so that llava_map_fn
    repositions it correctly.
    """
    return f"{GLOBAL_INSTRUCTION}\n<image>"


def build_gpt_output(label: str) -> str:
    """
    Build the structured GPT output from ONLY the Final diagnosis label.

    The evidence and explanation sections are auto-generated templates
    designed to teach the model visual reasoning during training.
    They are NOT used for accuracy scoring.

    Parameters
    ----------
    label : str
        Ground-truth label – one of KIRC, KIRP, KICH, N/A.

    Returns
    -------
    str
        Formatted output string for the 'gpt' conversation turn.
    """
    label = label.upper().strip()
    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label '{label}'. Must be one of {VALID_LABELS}."
        )

    # Auto-generate evidence: positive for the correct label, negative for others
    kirc_ev = _POSITIVE_EVIDENCE["KIRC"] if label == "KIRC" else _NEGATIVE_EVIDENCE["KIRC"]
    kirp_ev = _POSITIVE_EVIDENCE["KIRP"] if label == "KIRP" else _NEGATIVE_EVIDENCE["KIRP"]
    kich_ev = _POSITIVE_EVIDENCE["KICH"] if label == "KICH" else _NEGATIVE_EVIDENCE["KICH"]

    # Auto-generate presence scores
    kirc_score = "Positive" if label == "KIRC" else "Negative"
    kirp_score = "Positive" if label == "KIRP" else "Negative"
    kich_score = "Positive" if label == "KICH" else "Negative"

    # Auto-generate explanation
    explanation = _EXPLANATIONS[label]

    # ── Assemble output ──
    # Everything above "Final diagnosis" is visual reasoning (for training).
    # Only "Final diagnosis: <LABEL>" is the actual scored label.
    output = (
        f"KIRC evidence: {kirc_ev}\n"
        f"KIRP evidence: {kirp_ev}\n"
        f"KICH evidence: {kich_ev}\n"
        f"\n"
        f"KIRC: {kirc_score}\n"
        f"KIRP: {kirp_score}\n"
        f"KICH: {kich_score}\n"
        f"\n"
        f"Final diagnosis: {label}\n"
        f"Explanation: {explanation}"
    )
    return output


# ─────────────────────────────────────────────────────────────────────
# 3. Scoring – ONLY uses Final diagnosis
# ─────────────────────────────────────────────────────────────────────

def extract_final_diagnosis(text: str) -> Optional[str]:
    """
    Extract the Final diagnosis label from model output.
    This is the ONLY field used for accuracy scoring.

    Returns
    -------
    str or None
        Extracted label (KIRC/KIRP/KICH/N/A), or None if not found.
    """
    pattern = r"Final diagnosis:\s*(KIRC|KIRP|KICH|N/A)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def compute_accuracy(prediction_text: str, ground_truth_label: str) -> float:
    """
    Compare the extracted Final diagnosis against the ground truth.

    Returns
    -------
    float
        1.0 if correct, 0.0 if wrong or extraction failed.
    """
    predicted = extract_final_diagnosis(prediction_text)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth_label.upper().strip() else 0.0


# ─────────────────────────────────────────────────────────────────────
# 4. Data Conversion
# ─────────────────────────────────────────────────────────────────────

def load_label_map(label_map_path: str) -> Dict[str, str]:
    """
    Load a CSV with two columns: slide_id, label

    Example CSV:
        slide_id,label
        TCGA-A7-A0CJ-01Z-00-DX2,KIRC
        TCGA-AR-A255-01Z-00-DX1,KIRP
        TCGA-A7-A6VV-01Z-00-DX2,KICH
    """
    label_map = {}
    with open(label_map_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["slide_id"].strip()
            lbl = row["label"].strip().upper()
            if lbl not in VALID_LABELS:
                print(f"[WARN] Skipping unknown label '{lbl}' for {sid}")
                continue
            label_map[sid] = lbl
    return label_map


def build_entry(slide_id: str, image_file: str, label: str, idx: int) -> dict:
    """
    Build one training entry from a slide_id + label.

    The entire GPT output (evidence, scores, explanation) is auto-generated
    from the label alone. Only "Final diagnosis" is the real label.
    """
    return {
        "id": f"concept::{idx}",
        "image": [image_file],
        "conversations": [
            {
                "from": "human",
                "value": build_human_prompt(),
            },
            {
                "from": "gpt",
                "value": build_gpt_output(label=label),
            },
        ],
    }


def convert_from_label_map(
    label_map_path: str,
    output_path: str,
    image_dir: Optional[str] = None,
    image_ext: str = ".csv",
) -> None:
    """
    Generate a full training JSON from a label CSV.

    Parameters
    ----------
    label_map_path : str
        CSV with columns: slide_id, label
    output_path : str
        Where to write the output JSON.
    image_dir : str, optional
        Directory containing WSI feature CSVs. If provided, the image
        path will be prefixed (e.g. "WSI_feat/TCGA-xxx.csv").
    image_ext : str
        File extension for image features (default: ".csv").
    """
    label_map = load_label_map(label_map_path)
    print(f"Loaded {len(label_map)} labels from {label_map_path}")

    entries = []
    for idx, (slide_id, label) in enumerate(label_map.items()):
        image_file = f"{slide_id}{image_ext}"
        if image_dir:
            image_file = os.path.join(image_dir, image_file)

        entries.append(build_entry(slide_id, image_file, label, idx))

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    print(f"Generated {len(entries)} training entries -> {output_path}")


def convert_from_existing_json(
    input_path: str,
    output_path: str,
    label_map_path: str,
) -> None:
    """
    Convert an existing SlideChat VQA JSON, replacing the old conversations
    with the new concept-based format. Image paths are preserved.

    Requires a label_map CSV to know the Final diagnosis for each slide.
    """
    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {input_path}")

    label_map = load_label_map(label_map_path)
    print(f"Loaded {len(label_map)} labels from {label_map_path}")

    converted = []
    skipped = 0

    for idx, entry in enumerate(data):
        images = entry.get("image", [])
        if not images:
            skipped += 1
            continue

        # Extract slide_id from filename
        slide_id = os.path.splitext(os.path.basename(images[0]))[0]

        # Look up label
        label = label_map.get(slide_id)
        if label is None:
            # Try short TCGA ID (first 3 segments)
            short_id = "-".join(slide_id.split("-")[:3])
            label = label_map.get(short_id)

        if label is None:
            print(f"[WARN] No label for '{slide_id}', skipping.")
            skipped += 1
            continue

        converted.append(build_entry(slide_id, images[0], label, idx))

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    print(f"\nConversion complete:")
    print(f"  Converted: {len(converted)}")
    print(f"  Skipped:   {skipped}")
    print(f"  Output:    {output_path}")


# ─────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Stage 2 training data in Concept-Based Reasoning format. "
            "Input: label CSV (slide_id, label). "
            "The ONLY label is Final diagnosis. All evidence text is auto-generated."
        )
    )
    parser.add_argument(
        "--label_map", required=True,
        help="CSV file with columns: slide_id, label (KIRC/KIRP/KICH/N/A)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the output training JSON."
    )
    parser.add_argument(
        "--image_dir", default=None,
        help="Optional directory prefix for WSI feature CSVs."
    )
    parser.add_argument(
        "--input", default=None,
        help=(
            "Optional: existing SlideChat JSON to convert. "
            "If not provided, generates fresh entries from --label_map."
        )
    )
    args = parser.parse_args()

    if args.input:
        convert_from_existing_json(args.input, args.output, args.label_map)
    else:
        convert_from_label_map(args.label_map, args.output, args.image_dir)

    # --- Sanity check ---
    with open(args.output) as f:
        sample = json.load(f)
    if sample:
        gpt_text = sample[0]["conversations"][1]["value"]
        diag = extract_final_diagnosis(gpt_text)
        print(f"\n--- Sanity Check (first entry) ---")
        print(f"Final diagnosis: {diag}")
        print(f"Accuracy score:  {compute_accuracy(gpt_text, diag)}")


if __name__ == "__main__":
    main()
