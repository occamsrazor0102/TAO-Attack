"""
Gradio-based GUI for TAO-Attack.

Launch with:
    python gui.py
"""

import csv
import glob
import html
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import gradio as gr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_HARMFUL_DATASET_KEYWORDS = ("harm", "jailbreak", "attack", "advbench", "unsafe")
_NUMERIC_RE = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")
_MAX_RUN_LABEL_LEN = 16
_BYTES_PER_GIB = 1024 ** 3
_GUI_RUN_CONFIG_PATH = os.path.join(_ROOT_DIR, ".gui_attack_config.json")

_PRESETS = {
    "fast": {
        "cl_threshold": 0.9,
        "num_steps": 200,
        "batch_size": 128,
        "topk": 128,
        "temp": 0.6,
        "alpha": 0.2,
        "beta": 0.2,
    },
    "balanced": {
        "cl_threshold": 1.0,
        "num_steps": 1000,
        "batch_size": 256,
        "topk": 256,
        "temp": 0.5,
        "alpha": 0.2,
        "beta": 0.2,
    },
    "thorough": {
        "cl_threshold": 1.05,
        "num_steps": 2000,
        "batch_size": 384,
        "topk": 384,
        "temp": 0.4,
        "alpha": 0.25,
        "beta": 0.25,
    },
}

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_process: subprocess.Popen | None = None
_log_lines: list[str] = []
_log_lock = threading.Lock()
_run_started_at: float | None = None
_run_config: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_resolve(path: str) -> str:
    """Resolve *path* relative to the project root and return its real path."""
    resolved = os.path.realpath(os.path.join(_ROOT_DIR, path))
    if not resolved.startswith(_ROOT_DIR):
        raise ValueError(f"Path escapes project directory: {path}")
    return resolved


def _validate_numeric(value: str | int | float, name: str) -> str:
    """Return *value* as a string after verifying it looks numeric."""
    s = str(value)
    if not _NUMERIC_RE.match(s):
        raise ValueError(f"Invalid value for {name}: {s}")
    return s


def _looks_like_local_path(path: str) -> bool:
    if not path:
        return False
    return (
        path.startswith(".")
        or path.startswith("~")
        or os.path.isabs(path)
        or path.endswith((".pt", ".bin", ".safetensors"))
    )


def _validate_model_path(model_path: str) -> str:
    value = (model_path or "").strip()
    if not value:
        raise ValueError("Please provide a model path.")
    if value.startswith("-"):
        raise ValueError("Model path cannot start with '-'.")
    if any(ord(ch) < 32 for ch in value):
        raise ValueError("Model path contains control characters.")

    if _looks_like_local_path(value):
        resolved_local = _safe_resolve(os.path.expanduser(value))
        if not os.path.exists(resolved_local):
            raise ValueError(f"Local model path not found: `{model_path}`")
        return resolved_local

    if ".." in value or not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise ValueError("Invalid HuggingFace model id format.")
    return value


def _read_stream(stream):
    """Read lines from *stream* and append them to the shared log buffer."""
    for line in iter(stream.readline, ""):
        with _log_lock:
            _log_lines.append(line)
    stream.close()


def _build_attack_cmd(config_path: str) -> list[str]:
    """Build the attack command for source runs and packaged Windows runs."""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        attack_exe = os.path.join(exe_dir, "attack.exe")
        if os.path.exists(attack_exe):
            return [attack_exe, "--config_path", config_path]
        raise ValueError(
            f"Packaged GUI expects attack.exe next to gui executable: `{attack_exe}`"
        )

    attack_script = os.path.join(_ROOT_DIR, "attack.py")
    return [sys.executable, attack_script, "--config_path", config_path]


def _is_harmful_context(data_path: str, data: list[dict]) -> bool:
    context = f"{data_path} " + " ".join(
        f"{d.get('behavior', '')} {d.get('behaviour', '')} {d.get('target', '')}" for d in data[:20]
    )
    text = context.lower()
    return any(hint in text for hint in _HARMFUL_DATASET_KEYWORDS)


def _extract_result_entries(filepath: str) -> list[dict]:
    entries: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _final_step(entry: dict) -> dict:
    process = entry.get("process", [])
    if process and isinstance(process, list):
        return process[-1]
    return {}


def _summary_for_entries(name: str, entries: list[dict]) -> dict:
    total = len(entries)
    if total == 0:
        return {
            "file": name,
            "total": 0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_queries": 0.0,
            "avg_final_loss": 0.0,
        }

    successes = sum(1 for e in entries if e.get("Success"))
    final_iters = []
    final_losses = []
    for e in entries:
        step = _final_step(e)
        if "iteration" in step:
            try:
                final_iters.append(float(step["iteration"]))
            except (TypeError, ValueError):
                pass
        if "current_loss" in step:
            try:
                final_losses.append(float(step["current_loss"]))
            except (TypeError, ValueError):
                pass

    return {
        "file": name,
        "total": total,
        "successes": successes,
        "success_rate": (successes / total) * 100,
        "avg_queries": (sum(final_iters) / len(final_iters)) if final_iters else 0.0,
        "avg_final_loss": (sum(final_losses) / len(final_losses)) if final_losses else 0.0,
    }


def _infer_resume_index(save_folder: str) -> int:
    files = find_result_files(save_folder)
    max_bidx = -1
    for fpath in files:
        try:
            entries = _extract_result_entries(fpath)
        except OSError:
            continue
        for e in entries:
            try:
                max_bidx = max(max_bidx, int(e.get("bidx", -1)))
            except (TypeError, ValueError):
                pass
    return max_bidx + 1


def _render_svg_plot(summaries: list[dict], outpath: str):
    width = 800
    height = 300
    margin = 30
    bar_gap = 12
    n = max(len(summaries), 1)
    bar_width = max((width - (2 * margin) - (n - 1) * bar_gap) // n, 20)

    bars = []
    labels = []
    for i, s in enumerate(summaries):
        x = margin + i * (bar_width + bar_gap)
        h = int((s["success_rate"] / 100.0) * (height - 2 * margin))
        y = height - margin - h
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="#4f46e5" />')
        label = html.escape(os.path.basename(s["file"])[:_MAX_RUN_LABEL_LEN])
        labels.append(f'<text x="{x + bar_width / 2}" y="{height - 8}" font-size="10" text-anchor="middle">{label}</text>')
        labels.append(f'<text x="{x + bar_width / 2}" y="{max(14, y - 4)}" font-size="10" text-anchor="middle">{s["success_rate"]:.1f}%</text>')

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">\n
<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n
<line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#333\"/>\n
{''.join(bars)}\n
{''.join(labels)}\n
</svg>"""
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(svg)


# ---------------------------------------------------------------------------
# Attack data helpers
# ---------------------------------------------------------------------------

def load_attack_data(filepath: str) -> list[dict]:
    """Load attack behaviours from a JSON file."""
    if not filepath:
        return []
    try:
        filepath = _safe_resolve(filepath)
    except ValueError:
        return []
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        return []
    return []


def available_categories(data: list[dict]) -> list[str]:
    categories = set()
    for item in data:
        cat = item.get("category") or item.get("type") or item.get("source") or ""
        if isinstance(cat, str) and cat.strip():
            categories.add(cat.strip())
    return ["All"] + sorted(categories)


def filter_attack_data(
    data: list[dict],
    category: str,
    start_idx: int,
    end_idx: int,
    keyword: str,
) -> list[dict]:
    filtered = data

    if category and category != "All":
        filtered = [
            d for d in filtered
            if (d.get("category") or d.get("type") or d.get("source") or "") == category
        ]

    start = max(int(start_idx or 0), 0)
    end = int(end_idx or len(filtered))
    if end > len(filtered):
        end = len(filtered)
    if end < start:
        return []
    filtered = filtered[start:end]

    kw = (keyword or "").strip().lower()
    if kw:
        filtered = [
            d for d in filtered
            if kw in str(d.get("behavior", d.get("behaviour", ""))).lower()
            or kw in str(d.get("target", "")).lower()
            or kw in str(d.get("id", "")).lower()
        ]

    return filtered


def format_attack_table(data: list[dict]) -> list[list[str]]:
    """Return rows for the Gradio Dataframe component."""
    rows = []
    for idx, item in enumerate(data):
        rows.append([
            str(item.get("id", idx)),
            item.get("behaviour", item.get("behavior", "")),
            item.get("target", ""),
            item.get("category", item.get("type", item.get("source", ""))),
        ])
    return rows


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def find_result_files(folder: str) -> list[str]:
    """Glob for *.jsonl files under *folder*."""
    if not folder:
        return []
    try:
        folder = _safe_resolve(folder)
    except ValueError:
        return []
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.jsonl")))


def load_results(filepath: str) -> str:
    """Read a JSONL result file and return a formatted Markdown summary."""
    if not filepath:
        return "_No file selected._"
    try:
        filepath = _safe_resolve(filepath)
    except ValueError:
        return "_Invalid file path._"
    if not os.path.isfile(filepath):
        return "_No file selected._"

    entries = _extract_result_entries(filepath)
    if not entries:
        return "_File is empty or contains no valid JSON lines._"

    parts: list[str] = []
    for idx, entry in enumerate(entries):
        success = entry.get("Success", False)
        behaviour = entry.get("user_prompt", "N/A")
        completion = entry.get("Completion", "N/A")
        bidx = entry.get("bidx", idx)
        status = "✅ Success" if success else "❌ Failed"
        parts.append(
            f"### Behaviour #{bidx} — {status}\n"
            f"**Prompt:** {behaviour}\n\n"
            f"**Completion:** {completion}\n"
        )

        last = _final_step(entry)
        if last:
            parts.append(
                f"- Iterations: {last.get('iteration', '?')}\n"
                f"- Final loss: {last.get('current_loss', '?')}\n"
                f"- Suffix: `{last.get('current_suffix', '?')}`\n"
            )
        parts.append("---\n")

    summary = _summary_for_entries(os.path.basename(filepath), entries)
    header = (
        f"**Results summary:** {summary['successes']}/{summary['total']} successful "
        f"({summary['success_rate']:.1f}%)\n\n"
        f"- Avg queries: {summary['avg_queries']:.1f}\n"
        f"- Avg final loss: {summary['avg_final_loss']:.4f}\n\n"
        "---\n\n"
    )
    return header + "\n".join(parts)


def refresh_results(save_folder: str):
    """Return list of result files in the save folder."""
    files = find_result_files(save_folder)
    if not files:
        return gr.update(choices=[], value=None), "_No result files found._"
    return gr.update(choices=files, value=files[0]), load_results(files[0])


def show_result(filepath: str):
    """Load and display a single result file."""
    return load_results(filepath)


def summarize_history(folder: str):
    files = find_result_files(folder)
    if not files:
        return gr.update(value=[]), "_No result files found for history._", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    summaries = []
    rows = []
    for fpath in files:
        try:
            entries = _extract_result_entries(fpath)
        except OSError:
            continue
        summary = _summary_for_entries(fpath, entries)
        summaries.append(summary)
        rows.append([
            os.path.basename(fpath),
            summary["total"],
            summary["successes"],
            f"{summary['success_rate']:.1f}%",
            f"{summary['avg_queries']:.1f}",
            f"{summary['avg_final_loss']:.4f}",
        ])

    if not rows:
        return gr.update(value=[]), "_No readable JSONL result files found._", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    choices = [os.path.basename(s["file"]) for s in summaries]
    return (
        gr.update(value=rows),
        f"✅ Loaded {len(rows)} runs for comparison.",
        gr.update(choices=choices, value=choices[0]),
        gr.update(choices=choices, value=choices[min(1, len(choices) - 1)]),
    )


def compare_runs(folder: str, run_a: str, run_b: str):
    if not run_a or not run_b:
        return "_Select two runs to compare._"

    files = {os.path.basename(p): p for p in find_result_files(folder)}
    if run_a not in files or run_b not in files:
        return "_Selected runs are not available in this folder._"

    try:
        a = _summary_for_entries(run_a, _extract_result_entries(files[run_a]))
        b = _summary_for_entries(run_b, _extract_result_entries(files[run_b]))
    except OSError:
        return "_Failed to read selected run files._"

    diff_rate = b["success_rate"] - a["success_rate"]
    diff_queries = b["avg_queries"] - a["avg_queries"]
    diff_loss = b["avg_final_loss"] - a["avg_final_loss"]

    return (
        f"### Run Comparison\n"
        f"- **A:** `{run_a}`\n"
        f"- **B:** `{run_b}`\n\n"
        f"- Success rate: **{a['success_rate']:.1f}% → {b['success_rate']:.1f}%** ({diff_rate:+.1f} pp)\n"
        f"- Avg queries: **{a['avg_queries']:.1f} → {b['avg_queries']:.1f}** ({diff_queries:+.1f})\n"
        f"- Avg final loss: **{a['avg_final_loss']:.4f} → {b['avg_final_loss']:.4f}** ({diff_loss:+.4f})\n"
    )


def export_history(folder: str):
    files = find_result_files(folder)
    if not files:
        return "⚠️ No result files to export.", None, None, None

    summaries = []
    for fpath in files:
        try:
            entries = _extract_result_entries(fpath)
        except OSError:
            continue
        summaries.append(_summary_for_entries(fpath, entries))

    if not summaries:
        return "⚠️ No readable result files to export.", None, None, None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        resolved = _safe_resolve(folder)
    except ValueError:
        return "⚠️ Invalid export folder.", None, None, None

    export_dir = os.path.join(resolved, "exports")
    os.makedirs(export_dir, exist_ok=True)

    csv_path = os.path.join(export_dir, f"history_summary_{ts}.csv")
    json_path = os.path.join(export_dir, f"history_summary_{ts}.json")
    svg_path = os.path.join(export_dir, f"history_success_rate_{ts}.svg")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "total", "successes", "success_rate", "avg_queries", "avg_final_loss"])
        writer.writeheader()
        writer.writerows(summaries)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    _render_svg_plot(summaries, svg_path)

    return "✅ Exported CSV/JSON/SVG summary artifacts.", csv_path, json_path, svg_path


# ---------------------------------------------------------------------------
# Core callbacks
# ---------------------------------------------------------------------------

def apply_preset(profile: str):
    preset = _PRESETS.get((profile or "").lower())
    if not preset:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "⚠️ Unknown preset."
        )
    return (
        preset["cl_threshold"],
        preset["num_steps"],
        preset["batch_size"],
        preset["topk"],
        preset["temp"],
        preset["alpha"],
        preset["beta"],
        f"✅ Applied '{profile}' preset.",
    )


def refresh_data(data_path: str, category: str, start_idx: int, end_idx: int, keyword: str):
    """Reload attack data from the given JSON file."""
    data = load_attack_data(data_path)
    if not data:
        return gr.update(value=[]), f"⚠️ No data loaded from `{data_path}`", gr.update(choices=["All"], value="All")

    categories = available_categories(data)
    if category not in categories:
        category = "All"

    filtered = filter_attack_data(data, category, start_idx, end_idx, keyword)
    rows = format_attack_table(filtered)
    status = f"✅ Loaded {len(filtered)}/{len(data)} behaviours from `{data_path}`"
    return gr.update(value=rows), status, gr.update(choices=categories, value=category)


def _build_attack_config(
    model_path: str,
    save_folder: str,
    data_path: str,
    cl_threshold: float,
    num_steps: int,
    batch_size: int,
    topk: int,
    temp: float,
    alpha: float,
    beta: float,
    resume_run: bool,
    safe_mode: bool,
    confirm_harmful: bool,
) -> tuple[dict, int]:
    if not model_path:
        raise ValueError("Please provide a model path.")
    if not save_folder:
        raise ValueError("Please provide a save folder.")
    if not data_path:
        raise ValueError("Please provide a dataset path.")

    resolved_save = _safe_resolve(save_folder)
    resolved_data = _safe_resolve(data_path)
    if not os.path.exists(resolved_data):
        raise ValueError(f"Dataset file does not exist: `{data_path}`")

    data = load_attack_data(data_path)
    if not data:
        raise ValueError("Dataset is empty or invalid JSON list.")

    if safe_mode and _is_harmful_context(data_path, data) and not confirm_harmful:
        raise ValueError("Safe mode is enabled; please confirm launching against potentially harmful behavior data.")

    validated_model = _validate_model_path(model_path)

    start_bidx = 0
    if resume_run:
        start_bidx = _infer_resume_index(save_folder)

    attack_config = {
        "model_path": validated_model,
        "save_folder": resolved_save,
        "data_path": resolved_data,
        "cl_threshold": float(_validate_numeric(cl_threshold, "cl_threshold")),
        "num_steps": int(_validate_numeric(int(num_steps), "num_steps")),
        "batch_size": int(_validate_numeric(int(batch_size), "batch_size")),
        "topk": int(_validate_numeric(int(topk), "topk")),
        "temp": float(_validate_numeric(temp, "temp")),
        "alpha": float(_validate_numeric(alpha, "alpha")),
        "beta": float(_validate_numeric(beta, "beta")),
        "start_bidx": int(start_bidx),
    }
    return attack_config, len(data)


def dry_run_config(
    model_path: str,
    save_folder: str,
    data_path: str,
    cl_threshold: float,
    num_steps: int,
    batch_size: int,
    topk: int,
    temp: float,
    alpha: float,
    beta: float,
    resume_run: bool,
    safe_mode: bool,
    confirm_harmful: bool,
):
    try:
        attack_config, data_count = _build_attack_config(
            model_path,
            save_folder,
            data_path,
            cl_threshold,
            num_steps,
            batch_size,
            topk,
            temp,
            alpha,
            beta,
            resume_run,
            safe_mode,
            confirm_harmful,
        )
    except ValueError as exc:
        return f"⚠️ {exc}"

    try:
        cmd = _build_attack_cmd(_GUI_RUN_CONFIG_PATH)
    except ValueError as exc:
        return f"⚠️ {exc}"
    resume_msg = f"enabled from bidx={attack_config['start_bidx']}" if resume_run else "disabled"

    return (
        "✅ Dry run passed.\n\n"
        f"- Dataset size: {data_count}\n"
        f"- Resume: {resume_msg}\n"
        f"- Save folder: `{attack_config['save_folder']}`\n\n"
        "**Command preview**\n"
        f"`{' '.join(cmd)}`"
    )


def start_attack(
    model_path: str,
    save_folder: str,
    data_path: str,
    cl_threshold: float,
    num_steps: int,
    batch_size: int,
    topk: int,
    temp: float,
    alpha: float,
    beta: float,
    resume_run: bool,
    safe_mode: bool,
    confirm_harmful: bool,
):
    """Launch attack.py as a subprocess."""
    global _process, _run_started_at, _run_config

    if _process is not None and _process.poll() is None:
        return "⚠️ An attack is already running. Stop it first."

    validation_msg = dry_run_config(
        model_path,
        save_folder,
        data_path,
        cl_threshold,
        num_steps,
        batch_size,
        topk,
        temp,
        alpha,
        beta,
        resume_run,
        safe_mode,
        confirm_harmful,
    )
    if validation_msg.startswith("⚠️"):
        return validation_msg

    try:
        attack_config, _ = _build_attack_config(
            model_path,
            save_folder,
            data_path,
            cl_threshold,
            num_steps,
            batch_size,
            topk,
            temp,
            alpha,
            beta,
            resume_run,
            safe_mode,
            confirm_harmful,
        )
    except ValueError as exc:
        return f"⚠️ {exc}"

    with open(_GUI_RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(attack_config, f)

    try:
        cmd = _build_attack_cmd(_GUI_RUN_CONFIG_PATH)
    except ValueError as exc:
        return f"⚠️ {exc}"

    with _log_lock:
        _log_lines.clear()
        _log_lines.append(f"$ {' '.join(cmd)}\n")

    _process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=_ROOT_DIR,
    )

    threading.Thread(target=_read_stream, args=(_process.stdout,), daemon=True).start()

    _run_started_at = time.time()
    _run_config = {
        "num_steps": int(attack_config["num_steps"]),
        "start_bidx": int(attack_config["start_bidx"]),
        "data_path": data_path,
        "save_folder": save_folder,
    }

    return "🚀 Attack started."


def stop_attack():
    """Stop the running attack subprocess."""
    global _process
    if _process is None or _process.poll() is not None:
        return "ℹ️ No attack is currently running."
    _process.send_signal(signal.SIGTERM)
    try:
        _process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _process.kill()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _process = None
    with _log_lock:
        _log_lines.append("\n--- Attack stopped by user ---\n")
    return "🛑 Attack stopped."


def get_attack_status():
    """Return whether the attack is still running."""
    if _process is None:
        return "Idle"
    if _process.poll() is None:
        return "🟢 Running"
    code = _process.returncode
    return f"Finished (exit code {code})"


def _estimate_eta() -> str:
    if _run_started_at is None:
        return "N/A"

    with _log_lock:
        text = "".join(_log_lines[-300:])

    match = re.findall(r"Iteration\s+(\d+)", text)
    if not match:
        return "N/A"

    try:
        current_iter = int(match[-1])
        total_steps = int(_run_config.get("num_steps", 0))
        if current_iter <= 0 or total_steps <= 0:
            return "N/A"
        elapsed = max(time.time() - _run_started_at, 0.0)
        sec_per_iter = elapsed / current_iter
        remaining = max(total_steps - current_iter, 0)
        eta_seconds = int(sec_per_iter * remaining)
        return f"~{eta_seconds}s"
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"


def monitor_resources():
    elapsed = 0
    if _run_started_at is not None:
        elapsed = int(time.time() - _run_started_at)

    gpu_used = "N/A"
    gpu_total = "N/A"
    try:
        import torch  # lazy import

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / _BYTES_PER_GIB
            total = torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GIB
            gpu_used = f"{used:.2f} GiB"
            gpu_total = f"{total:.2f} GiB"
    except Exception:
        pass

    return (
        f"- Runtime: **{elapsed}s**\n"
        f"- ETA: **{_estimate_eta()}**\n"
        f"- GPU memory: **{gpu_used} / {gpu_total}**\n"
        f"- Status: **{get_attack_status()}**"
    )


def poll_logs():
    """Return the current log text."""
    with _log_lock:
        text = "".join(_log_lines[-500:])
    return text


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    default_data_path = os.path.join("data", "advbench", "igcg_ori.json")

    with gr.Blocks(title="TAO-Attack GUI") as demo:
        gr.Markdown("# 🔬 TAO-Attack GUI\nConfigure, run, and inspect TAO-Attack experiments.")

        # ── Tab 1: Configuration & Launch ──────────────────────────────
        with gr.Tab("⚙️ Configuration & Launch"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Model, Data & Output")
                    model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="/path/to/model or hf/model-id",
                        info="Local model path or HuggingFace model id",
                    )
                    data_path = gr.Textbox(
                        label="Dataset Path",
                        value=default_data_path,
                        info="JSON list of attack behaviors/targets",
                    )
                    save_folder = gr.Textbox(
                        label="Save Folder",
                        placeholder="results/",
                        value="results",
                        info="Directory for output JSONL files",
                    )

                with gr.Column(scale=3):
                    gr.Markdown("### Hyperparameters")
                    with gr.Row():
                        preset = gr.Dropdown(label="Preset", choices=["fast", "balanced", "thorough"], value="balanced")
                        preset_btn = gr.Button("Apply Preset")
                    with gr.Row():
                        cl_threshold = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="CL Threshold (τ)")
                        num_steps = gr.Number(value=1000, label="Num Steps", precision=0)
                    with gr.Row():
                        batch_size = gr.Number(value=256, label="Batch Size", precision=0)
                        topk = gr.Number(value=256, label="Top-K", precision=0)
                    with gr.Row():
                        temp = gr.Slider(0.01, 2.0, value=0.5, step=0.05, label="Temperature (γ)")
                        alpha = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Alpha (α)")
                        beta = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Beta (β)")

            with gr.Row():
                resume_run = gr.Checkbox(label="Resume from previous outputs", value=False)
                safe_mode = gr.Checkbox(label="Safe mode (require confirmation for harmful datasets)", value=True)
                confirm_harmful = gr.Checkbox(label="I confirm I want to run potentially harmful-behavior data", value=False)

            with gr.Row():
                dry_run_btn = gr.Button("🧪 Dry Run / Smoke Test")
                start_btn = gr.Button("🚀 Start Attack", variant="primary")
                stop_btn = gr.Button("🛑 Stop Attack", variant="stop")

            launch_msg = gr.Textbox(label="Message", interactive=False, lines=6)

            preset_btn.click(
                fn=apply_preset,
                inputs=[preset],
                outputs=[cl_threshold, num_steps, batch_size, topk, temp, alpha, beta, launch_msg],
            )
            dry_run_btn.click(
                fn=dry_run_config,
                inputs=[
                    model_path,
                    save_folder,
                    data_path,
                    cl_threshold,
                    num_steps,
                    batch_size,
                    topk,
                    temp,
                    alpha,
                    beta,
                    resume_run,
                    safe_mode,
                    confirm_harmful,
                ],
                outputs=launch_msg,
            )
            start_btn.click(
                fn=start_attack,
                inputs=[
                    model_path,
                    save_folder,
                    data_path,
                    cl_threshold,
                    num_steps,
                    batch_size,
                    topk,
                    temp,
                    alpha,
                    beta,
                    resume_run,
                    safe_mode,
                    confirm_harmful,
                ],
                outputs=launch_msg,
            )
            stop_btn.click(fn=stop_attack, outputs=launch_msg)

        # ── Tab 2: Attack Data Browser ─────────────────────────────────
        with gr.Tab("📋 Attack Data"):
            with gr.Row():
                data_path_view = gr.Textbox(
                    label="Data JSON Path",
                    value=default_data_path,
                    info="Path to the behaviors JSON file",
                )
                load_data_btn = gr.Button("🔄 Load")

            with gr.Row():
                category_filter = gr.Dropdown(label="Category", choices=["All"], value="All")
                idx_start = gr.Number(label="Start Index", value=0, precision=0)
                idx_end = gr.Number(label="End Index", value=50, precision=0)
                keyword = gr.Textbox(label="Keyword", placeholder="filter by prompt/target")

            data_status = gr.Textbox(label="Status", interactive=False)
            data_table = gr.Dataframe(
                headers=["ID", "Behaviour", "Target", "Category"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            load_data_btn.click(
                fn=refresh_data,
                inputs=[data_path_view, category_filter, idx_start, idx_end, keyword],
                outputs=[data_table, data_status, category_filter],
            )
            category_filter.change(
                fn=refresh_data,
                inputs=[data_path_view, category_filter, idx_start, idx_end, keyword],
                outputs=[data_table, data_status, category_filter],
            )
            idx_start.change(
                fn=refresh_data,
                inputs=[data_path_view, category_filter, idx_start, idx_end, keyword],
                outputs=[data_table, data_status, category_filter],
            )
            idx_end.change(
                fn=refresh_data,
                inputs=[data_path_view, category_filter, idx_start, idx_end, keyword],
                outputs=[data_table, data_status, category_filter],
            )
            keyword.submit(
                fn=refresh_data,
                inputs=[data_path_view, category_filter, idx_start, idx_end, keyword],
                outputs=[data_table, data_status, category_filter],
            )

            demo.load(
                fn=refresh_data,
                inputs=[data_path_view, category_filter, idx_start, idx_end, keyword],
                outputs=[data_table, data_status, category_filter],
            )

        # ── Tab 3: Live Logs ───────────────────────────────────────────
        with gr.Tab("📜 Live Logs"):
            log_box = gr.Textbox(
                label="Attack Logs (last 500 lines)",
                lines=22,
                max_lines=500,
                interactive=False,
            )
            resource_md = gr.Markdown("- Runtime: **0s**\n- ETA: **N/A**\n- GPU memory: **N/A / N/A**\n- Status: **Idle**")

            with gr.Row():
                refresh_log_btn = gr.Button("🔄 Refresh Logs")
                status_btn = gr.Button("📡 Check Status")
                status_display = gr.Textbox(label="Process Status", interactive=False)

            refresh_log_btn.click(fn=poll_logs, outputs=log_box)
            status_btn.click(fn=get_attack_status, outputs=status_display)

            log_timer = gr.Timer(value=3)
            log_timer.tick(fn=poll_logs, outputs=log_box)
            monitor_timer = gr.Timer(value=5)
            monitor_timer.tick(fn=monitor_resources, outputs=resource_md)

        # ── Tab 4: Results Viewer ──────────────────────────────────────
        with gr.Tab("📊 Results"):
            with gr.Row():
                results_folder = gr.Textbox(
                    label="Results Folder",
                    value="results",
                    info="Same as the save folder used during the attack",
                )
                refresh_results_btn = gr.Button("🔄 Refresh")

            result_file_dropdown = gr.Dropdown(label="Result File", choices=[], interactive=True)
            results_display = gr.Markdown("_Click Refresh to load results._")

            refresh_results_btn.click(
                fn=refresh_results,
                inputs=results_folder,
                outputs=[result_file_dropdown, results_display],
            )
            result_file_dropdown.change(fn=show_result, inputs=result_file_dropdown, outputs=results_display)

        # ── Tab 5: Run History & Export ───────────────────────────────
        with gr.Tab("📈 Run History & Export"):
            with gr.Row():
                history_folder = gr.Textbox(label="History Folder", value="results")
                history_refresh_btn = gr.Button("🔄 Load History")

            history_status = gr.Textbox(label="History Status", interactive=False)
            history_table = gr.Dataframe(
                headers=["File", "Total", "Successes", "Success Rate", "Avg Queries", "Avg Final Loss"],
                datatype=["str", "number", "number", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                run_a = gr.Dropdown(label="Compare Run A", choices=[])
                run_b = gr.Dropdown(label="Compare Run B", choices=[])
                compare_btn = gr.Button("Compare")

            comparison_md = gr.Markdown("_Load history and choose runs to compare._")

            with gr.Row():
                export_btn = gr.Button("⬇️ Export CSV/JSON/Plot")
                export_msg = gr.Textbox(label="Export Status", interactive=False)

            with gr.Row():
                export_csv = gr.File(label="CSV")
                export_json = gr.File(label="JSON")
                export_plot = gr.File(label="Plot (SVG)")

            history_refresh_btn.click(
                fn=summarize_history,
                inputs=[history_folder],
                outputs=[history_table, history_status, run_a, run_b],
            )
            compare_btn.click(
                fn=compare_runs,
                inputs=[history_folder, run_a, run_b],
                outputs=[comparison_md],
            )
            export_btn.click(
                fn=export_history,
                inputs=[history_folder],
                outputs=[export_msg, export_csv, export_json, export_plot],
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
