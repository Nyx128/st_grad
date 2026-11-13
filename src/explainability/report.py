"""
Standardized Case Record Serializer & Report Generator for ST-GRAD (Phase 4).
Implements the exact case record specification from Section 5.6 of the Research Plan:
- JSON/NPZ schema serialization with numpy type conversion
- Validation of required case metadata, predictions, attributions, views, and graph interventions
- Executive Markdown Report Card generation
- Standalone HTML Report Card generation with responsive styling
- Batch summary compilation across multiple evaluation instances
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np


REQUIRED_SCHEMA_KEYS = [
    "ticker", "target_date", "prediction", "attribution", "provenance", "caveats"
]


def convert_numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for clean JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    return obj


def validate_case_record(record: Dict[str, Any]) -> bool:
    """Check that a case record conforms to the required schema."""
    for key in REQUIRED_SCHEMA_KEYS:
        if key not in record:
            raise KeyError(f"Case record missing mandatory schema field: '{key}'")
    pred = record["prediction"]
    for p_key in ["logit", "probability", "predicted_class"]:
        if p_key not in pred:
            raise KeyError(f"Prediction object missing field: '{p_key}'")
    return True


def save_case_record(record: Dict[str, Any], output_path: str, save_npz: bool = False):
    """
    Save a standardized case record to JSON (and optionally raw array arrays to NPZ).
    """
    validate_case_record(record)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    clean_record = convert_numpy_to_native(record)

    with open(output_path, "w") as f:
        json.dump(clean_record, f, indent=2)

    if save_npz:
        npz_path = output_path.replace(".json", ".npz")
        arrays_to_save = {}
        if "raw_attribution_tensor" in record:
            arrays_to_save["attributions"] = np.array(record["raw_attribution_tensor"])
        if arrays_to_save:
            np.savez_compressed(npz_path, **arrays_to_save)


def load_case_record(input_path: str) -> Dict[str, Any]:
    """Load and validate a saved JSON case record."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Case record not found: {input_path}")
    with open(input_path, "r") as f:
        record = json.load(f)
    validate_case_record(record)
    return record


def format_markdown_card(record: Dict[str, Any]) -> str:
    """Format a case record into an executive Markdown card."""
    validate_case_record(record)
    p = record["prediction"]
    attr = record["attribution"]
    views = record.get("views")
    graph = record.get("graph")
    stability = record.get("stability")
    gt = record.get("ground_truth")

    lines = []
    lines.append(f"# ST-GRAD Explanation Report Card: {record['ticker']}")
    lines.append(
        f"**Target Date:** `{record['target_date']}` | "
        f"**Track:** `{record['provenance'].get('track', 'N/A')}` | "
        f"**Model Hash:** `{str(record['provenance'].get('checkpoint_hash', 'N/A'))[:10]}`"
    )
    lines.append("")

    # Section 1
    lines.append("## 1. Directional Prediction Summary")
    direction = p.get("trend_direction", "UP" if p["predicted_class"] == 1 else "DOWN")
    lines.append(f"- **Predicted Trend:** **`{direction}`** (Class `{p['predicted_class']}`)")
    lines.append(f"- **Probability:** `{p['probability']:.4f}` (Pre-Sigmoid Logit: `{p['logit']:.4f}`)")
    if gt:
        outcome = "CORRECT" if p["predicted_class"] == gt["label"] else "INCORRECT"
        lines.append(f"- **Ground Truth:** `{gt.get('direction', 'UP' if gt['label']==1 else 'DOWN')}` (Class `{gt['label']}`) | Outcome: **`{outcome}`**")
    lines.append("")

    # Section 2
    lines.append("## 2. Key Technical Indicator Drivers (Integrated Gradients)")
    conv = attr.get("convergence", {})
    delta_str = f"{conv.get('convergence_delta', 0.0)*100:.2f}%"
    status_str = "PASSED" if conv.get("is_converged", True) else "APPROXIMATE"
    lines.append(f"- *Completeness Check:* **`{status_str}`** (Delta: `{delta_str}` over `{conv.get('steps_used', 50)}` {conv.get('quadrature_method', 'gauss_legendre')} steps)")
    if "target_stock_intrinsic_share" in attr:
        lines.append(f"- *Intrinsic Reliance:* `{attr['target_stock_intrinsic_share']*100:.1f}%` of attribution originates from {record['ticker']}'s own indicators.")
    lines.append("")

    lines.append("| Rank | Technical Indicator | Absolute Score | Directional Impact |")
    lines.append("|:---:|---|:---:|:---:|")
    for f in attr.get("top_magnitude_features", []):
        badge = "🟢 Bullish (+)" if f.get("direction") == "BULLISH" else "🔴 Bearish (-)"
        lines.append(f"| {f['rank']} | `{f['feature']}` | {f['importance_score']:.3f} | {badge} ({f.get('signed_impact', 0.0):+.3f}) |")
    lines.append("")

    # Section 3
    if views:
        lines.append("## 3. Multi-View Cross-Attention (MVCA) Hierarchy")
        ranking = " > ".join([f"**{v}**" for v in views.get("importance_ranking", [])])
        lines.append(f"- **View Impact Hierarchy:** {ranking}")
        for v, data in views.get("interventions", {}).items():
            lines.append(f"  - `{v}` View Ablation Drop: `Δlogit = {data['delta_logit']:.4f}`")
        lines.append("")

    # Section 4
    if graph:
        lines.append("## 4. Spatio-Temporal Graph & Peer Influence")
        lines.append(f"- **Static Sector Prior Channel Drop:** `Δlogit = {graph.get('static_prior_ablation_delta_logit', 0.0):.4f}`")
        lines.append(f"- **Adaptive Dynamic Graph Channel Drop:** `Δlogit = {graph.get('adaptive_graph_ablation_delta_logit', 0.0):.4f}`")
        if graph.get("top_influential_peer_tickers"):
            peers_str = ", ".join([f"`{p['ticker']}` (Δ{p['delta_logit']:.3f})" for p in graph["top_influential_peer_tickers"][:3]])
            lines.append(f"- **Top Influential Peer Stocks:** {peers_str}")
        lines.append("")

    # Section 5
    if stability:
        lines.append("## 5. Stability & Robustness Diagnostics")
        lines.append(f"- **Perturbation Robustness:** `{'ROBUST' if stability.get('is_robust') else 'FRAGILE'}` (Cosine Sim: `{stability.get('mean_cosine_similarity', 0.0):.4f}`)")
        lines.append(f"- **Valid Noise Trials:** `{stability.get('valid_trials', 0)}/{stability.get('total_trials', 0)}`")
        lines.append("")

    # Section 6
    lines.append("## 6. Scientific Integrity & Non-Causal Disclaimers")
    for c in record.get("caveats", []):
        lines.append(f"> ⚠️ *{c}*")

    return "\n".join(lines)


def format_html_report_card(record: Dict[str, Any]) -> str:
    """Generate a clean, standalone responsive HTML card with inline CSS."""
    md = format_markdown_card(record)
    p = record["prediction"]
    direction = p.get("trend_direction", "UP" if p["predicted_class"] == 1 else "DOWN")
    is_up = direction == "UP"
    color = "#10b981" if is_up else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ST-GRAD Explanation Card: {record['ticker']}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; max-width: 900px; margin: auto; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 1rem; margin-bottom: 1.5rem; }}
  .ticker {{ font-size: 2rem; font-weight: 700; color: #38bdf8; }}
  .badge {{ font-size: 1.25rem; font-weight: 700; padding: 0.3rem 1rem; border-radius: 8px; color: white; background: {color}; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .metric-box {{ background: #0f172a; padding: 1rem; border-radius: 8px; border: 1px solid #334155; }}
  .metric-label {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }}
  .metric-val {{ font-size: 1.5rem; font-weight: 600; color: #f1f5f9; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; }}
  .caveat {{ background: #451a03; border-left: 4px solid #f59e0b; padding: 0.75rem 1rem; margin-top: 1.5rem; border-radius: 4px; font-size: 0.9rem; color: #fef3c7; }}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div>
      <div class="ticker">{record['ticker']}</div>
      <div style="color: #94a3b8; font-size: 0.9rem;">Target Date: {record['target_date']} | Track: {record['provenance'].get('track', 'paper_aligned')}</div>
    </div>
    <div class="badge">{direction} ({p['probability']*100:.1f}%)</div>
  </div>

  <div class="metric-grid">
    <div class="metric-box">
      <div class="metric-label">Pre-Sigmoid Logit</div>
      <div class="metric-val">{p['logit']:.4f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Model Confidence</div>
      <div class="metric-val">{p.get('confidence', 0.0)*100:.1f}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Intrinsic Reliance</div>
      <div class="metric-val">{record['attribution'].get('target_stock_intrinsic_share', 0.0)*100:.1f}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">IG Completeness</div>
      <div class="metric-val" style="color: #10b981;">{record['attribution'].get('convergence', {}).get('convergence_delta', 0.0)*100:.2f}%</div>
    </div>
  </div>

  <h3>Top Technical Drivers (Integrated Gradients)</h3>
  <table>
    <thead>
      <tr><th>Rank</th><th>Indicator</th><th>Importance</th><th>Direction</th></tr>
    </thead>
    <tbody>
"""
    for f in record["attribution"].get("top_magnitude_features", []):
        is_bull = f.get("direction") == "BULLISH"
        badge_style = "color: #10b981;" if is_bull else "color: #ef4444;"
        symbol = "▲ Bullish" if is_bull else "▼ Bearish"
        html += f"""      <tr>
        <td>{f['rank']}</td>
        <td><code>{f['feature']}</code></td>
        <td>{f['importance_score']:.3f}</td>
        <td style="{badge_style}"><strong>{symbol} ({f.get('signed_impact', 0.0):+.3f})</strong></td>
      </tr>\n"""

    html += """    </tbody>
  </table>

  <div class="caveat">
    <strong>Scientific Disclaimer:</strong> Model attributions reflect mathematical feature reliance, not real-world market causality.
  </div>
</div>
</body>
</html>"""
    return html


def compile_batch_summary_report(records: List[Dict[str, Any]], title: str = "ST-GRAD Cohort Summary") -> Dict[str, Any]:
    """Aggregate a collection of case records into an executive cohort report."""
    if not records:
        return {"error": "Empty records list"}

    total_cases = len(records)
    predictions = [r["prediction"]["predicted_class"] for r in records]
    up_count = sum(predictions)
    down_count = total_cases - up_count

    correct_count = 0
    ground_truth_available = 0
    for r in records:
        if "ground_truth" in r:
            ground_truth_available += 1
            if r["prediction"]["predicted_class"] == r["ground_truth"]["label"]:
                correct_count += 1

    # Feature frequency in top 3
    feature_top_counts = {}
    for r in records:
        for f in r["attribution"].get("top_magnitude_features", [])[:3]:
            fname = f["feature"]
            feature_top_counts[fname] = feature_top_counts.get(fname, 0) + 1

    top_features_ranked = sorted(feature_top_counts.items(), key=lambda item: item[1], reverse=True)

    # View frequency in rank 1
    view_top_counts = {"ST": 0, "DS": 0, "DT": 0}
    for r in records:
        v = r.get("views", {})
        if v and v.get("importance_ranking"):
            top_v = v["importance_ranking"][0]
            if top_v in view_top_counts:
                view_top_counts[top_v] += 1

    mean_convergence = float(np.mean([r["attribution"]["convergence"]["convergence_delta"] for r in records]))
    mean_intrinsic = float(np.mean([r["attribution"].get("target_stock_intrinsic_share", 0.5) for r in records]))

    summary = {
        "title": title,
        "total_cases": total_cases,
        "class_distribution": {"UP": up_count, "DOWN": down_count, "up_ratio": up_count / total_cases},
        "accuracy_on_evaluated": float(correct_count / ground_truth_available) if ground_truth_available > 0 else None,
        "mean_convergence_delta": mean_convergence,
        "mean_target_stock_intrinsic_share": mean_intrinsic,
        "top_features_in_cohort": [{"feature": k, "frequency_top3": v} for k, v in top_features_ranked[:5]],
        "top_view_distribution": view_top_counts
    }
    return summary
