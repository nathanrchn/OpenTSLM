# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""Scores test_predictions.jsonl with the paper's parsers in evaluation/opentslm (same F1/accuracy logic)."""

import contextlib
import importlib.util
import io
import json
import os
import sys

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation", "opentslm")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EVAL_DIR, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_tsqa(path):
    # Same logic as evaluation/opentslm/tsqa/parse_predictions.py (compare the first 3 chars, e.g. "(a)")
    har = _load("har_parser", "parse_predictions.py")
    points = []
    for e in _read(path):
        gen, gold = e["generated"].strip()[:3], e["gold"].strip()[:3]
        r = har.calculate_f1_score(gen, gold)
        points.append({"accuracy": gen == gold, **r})
    allowed = {p["ground_truth_normalized"] for p in points}
    f1 = har.calculate_f1_stats(points, allowed_labels=allowed)
    return {"n": len(points), "accuracy": 100 * sum(p["accuracy"] for p in points) / len(points),
            "macro_f1": 100 * f1["macro_f1"]}


def score_har(path):
    # Same logic as parse_rtf_jsonl in evaluation/opentslm/parse_predictions.py, but reading the JSON
    # properly (its regex extraction truncates generations that contain a double quote).
    har = _load("har_parser", "parse_predictions.py")
    points = []
    for e in _read(path):
        pred = har.extract_answer(e["generated"]).replace("<eos>", "")
        gold = har.extract_answer(e["gold"]).replace("<eos>", "")
        r = har.calculate_f1_score(pred, gold)
        points.append({"accuracy": pred == gold, **r})
    allowed = set(har.SUPPORTED_LABELS)
    acc = har.calculate_accuracy_stats(points)
    f1 = har.calculate_f1_stats(points, allowed_labels=allowed)
    return {"n": len(points), "accuracy": acc["accuracy_percentage"], "macro_f1": 100 * f1["macro_f1"],
            "n_out_of_label_set": sum(p["prediction_normalized"] not in allowed for p in points)}


def score_sleep(path):
    sl = _load("sleep_parser", "sleep/parse_sleep_cot_data.py")
    clean = path.replace(".jsonl", ".clean.jsonl")
    with contextlib.redirect_stdout(io.StringIO()):
        points = sl.parse_sleep_cot_jsonl(path, clean)
    acc = sl.calculate_accuracy_stats(points)
    f1 = sl.calculate_f1_stats(points)
    return {"n": len(points), "accuracy": acc["accuracy_percentage"], "macro_f1": 100 * f1["macro_f1"]}


def score_ecg(path):
    ecg = _load("ecg_parser", "ecg_qa_cot/parse_ecg_qa_cot_data.py")
    conv = path.replace(".jsonl", ".ecgfmt.jsonl")
    with open(conv, "w", encoding="utf-8") as f:
        for e in _read(path):
            f.write(json.dumps({"generated_answer": e["generated"], "target_answer": e["gold"],
                                "template_id": e.get("template_id"), "ecg_id": e.get("ecg_id", "None")}) + "\n")
    clean = path.replace(".jsonl", ".clean.jsonl")
    with contextlib.redirect_stdout(io.StringIO()):
        points = ecg.parse_ecg_qa_cot_jsonl(conv, clean)
    acc = ecg.calculate_accuracy_stats(points)
    f1 = ecg.calculate_template_f1_stats(points)["overall"]
    return {"n": len(points), "accuracy": acc.get("accuracy_percentage", acc.get("accuracy")),
            "macro_f1_weighted": 100 * f1["macro_f1_weighted"], "macro_f1_unweighted": 100 * f1["macro_f1"]}


def score(task, path):
    return {"tsqa": score_tsqa, "har": score_har, "sleep": score_sleep, "ecg": score_ecg}[task](path)


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2]), indent=2))
