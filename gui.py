"""
Gradio-based GUI for TAO-Attack.

Launch with:
    python gui.py
"""

import glob
import json
import os
import re
import signal
import subprocess
import threading

import gradio as gr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_process: subprocess.Popen | None = None
_log_lines: list[str] = []
_log_lock = threading.Lock()


def _safe_resolve(path: str) -> str:
    """Resolve *path* relative to the project root and return its real path.

    Raises ``ValueError`` if the resolved path escapes the project tree.
    """
    resolved = os.path.realpath(os.path.join(_ROOT_DIR, path))
    if not resolved.startswith(_ROOT_DIR):
        raise ValueError(f"Path escapes project directory: {path}")
    return resolved


def _read_stream(stream):
    """Read lines from *stream* and append them to the shared log buffer."""
    for line in iter(stream.readline, ""):
        with _log_lock:
            _log_lines.append(line)
    stream.close()


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
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def format_attack_table(data: list[dict]) -> list[list[str]]:
    """Return rows for the Gradio Dataframe component."""
    rows = []
    for item in data:
        rows.append([
            str(item.get("id", "")),
            item.get("behaviour", item.get("behavior", "")),
            item.get("target", ""),
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

    entries: list[dict] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

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
        # Show last recorded step info
        process = entry.get("process", [])
        if process:
            last = process[-1]
            parts.append(
                f"- Iterations: {last.get('iteration', '?')}\n"
                f"- Final loss: {last.get('current_loss', '?')}\n"
                f"- Suffix: `{last.get('current_suffix', '?')}`\n"
            )
        parts.append("---\n")

    total = len(entries)
    successes = sum(1 for e in entries if e.get("Success"))
    header = (
        f"**Results summary:** {successes}/{total} successful "
        f"({successes / total * 100:.1f}%)\n\n---\n\n"
    )
    return header + "\n".join(parts)


# ---------------------------------------------------------------------------
# Core callbacks
# ---------------------------------------------------------------------------

def refresh_data(data_path: str):
    """Reload attack data from the given JSON file."""
    data = load_attack_data(data_path)
    if not data:
        return gr.update(value=[]), f"⚠️  No data loaded from `{data_path}`"
    rows = format_attack_table(data)
    return gr.update(value=rows), f"✅ Loaded {len(data)} behaviours from `{data_path}`"


_NUMERIC_RE = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")


def _validate_numeric(value: str, name: str) -> str:
    """Return *value* as a string after verifying it looks numeric."""
    s = str(value)
    if not _NUMERIC_RE.match(s):
        raise ValueError(f"Invalid value for {name}: {s}")
    return s


def start_attack(
    model_path: str,
    save_folder: str,
    cl_threshold: float,
    num_steps: int,
    batch_size: int,
    topk: int,
    temp: float,
    alpha: float,
    beta: float,
):
    """Launch attack.py as a subprocess."""
    global _process
    if _process is not None and _process.poll() is None:
        return "⚠️  An attack is already running. Stop it first."

    if not model_path:
        return "⚠️  Please provide a model path."
    if not save_folder:
        return "⚠️  Please provide a save folder."

    # Validate save_folder stays within project directory
    try:
        resolved_save = _safe_resolve(save_folder)
    except ValueError:
        return "⚠️  Save folder must be inside the project directory."

    # Validate numeric parameters
    try:
        cmd = [
            "python", "attack.py",
            "--model_path", str(model_path),
            "--save_folder", resolved_save,
            "--cl_threshold", _validate_numeric(cl_threshold, "cl_threshold"),
            "--num_steps", _validate_numeric(int(num_steps), "num_steps"),
            "--batch_size", _validate_numeric(int(batch_size), "batch_size"),
            "--topk", _validate_numeric(int(topk), "topk"),
            "--temp", _validate_numeric(temp, "temp"),
            "--alpha", _validate_numeric(alpha, "alpha"),
            "--beta", _validate_numeric(beta, "beta"),
        ]
    except ValueError as exc:
        return f"⚠️  {exc}"

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

    t = threading.Thread(target=_read_stream, args=(_process.stdout,), daemon=True)
    t.start()

    return "🚀 Attack started."


def stop_attack():
    """Stop the running attack subprocess."""
    global _process
    if _process is None or _process.poll() is not None:
        return "ℹ️  No attack is currently running."
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


def poll_logs():
    """Return the current log text."""
    with _log_lock:
        text = "".join(_log_lines[-500:])
    return text


def refresh_results(save_folder: str):
    """Return list of result files in the save folder."""
    files = find_result_files(save_folder)
    if not files:
        return gr.update(choices=[], value=None), "_No result files found._"
    return gr.update(choices=files, value=files[0]), load_results(files[0])


def show_result(filepath: str):
    """Load and display a single result file."""
    return load_results(filepath)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    default_data_path = os.path.join("data", "advbench", "igcg.json")

    with gr.Blocks(title="TAO-Attack GUI") as demo:
        gr.Markdown("# 🔬 TAO-Attack GUI\nConfigure, run, and inspect TAO-Attack experiments.")

        # ── Tab 1: Configuration & Launch ──────────────────────────────
        with gr.Tab("⚙️ Configuration & Launch"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Model & Output")
                    model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="/path/to/model",
                        info="Local path or HuggingFace model id",
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
                start_btn = gr.Button("🚀 Start Attack", variant="primary")
                stop_btn = gr.Button("🛑 Stop Attack", variant="stop")
                status_box = gr.Textbox(label="Status", interactive=False)

            launch_msg = gr.Textbox(label="Message", interactive=False)

            start_btn.click(
                fn=start_attack,
                inputs=[model_path, save_folder, cl_threshold, num_steps, batch_size, topk, temp, alpha, beta],
                outputs=launch_msg,
            )
            stop_btn.click(fn=stop_attack, outputs=launch_msg)

        # ── Tab 2: Attack Data Browser ─────────────────────────────────
        with gr.Tab("📋 Attack Data"):
            with gr.Row():
                data_path = gr.Textbox(
                    label="Data JSON Path",
                    value=default_data_path,
                    info="Path to the behaviours JSON file",
                )
                load_data_btn = gr.Button("🔄 Load")

            data_status = gr.Textbox(label="Status", interactive=False)
            data_table = gr.Dataframe(
                headers=["ID", "Behaviour", "Target"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            load_data_btn.click(fn=refresh_data, inputs=data_path, outputs=[data_table, data_status])
            # Auto-load on start
            demo.load(fn=refresh_data, inputs=data_path, outputs=[data_table, data_status])

        # ── Tab 3: Live Logs ───────────────────────────────────────────
        with gr.Tab("📜 Live Logs"):
            log_box = gr.Textbox(
                label="Attack Logs (last 500 lines)",
                lines=25,
                max_lines=500,
                interactive=False,
            )
            with gr.Row():
                refresh_log_btn = gr.Button("🔄 Refresh Logs")
                status_btn = gr.Button("📡 Check Status")
                status_display = gr.Textbox(label="Process Status", interactive=False)

            refresh_log_btn.click(fn=poll_logs, outputs=log_box)
            status_btn.click(fn=get_attack_status, outputs=status_display)

            # Auto-refresh logs every 3 seconds while the tab is open
            log_timer = gr.Timer(value=3)
            log_timer.tick(fn=poll_logs, outputs=log_box)

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

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
