# TAO-Attack

This repository contains the official implementation of **TAO-Attack**, a novel optimization-based jailbreak attack for large language models (LLMs).  
The attack integrates a two-stage loss function and a direction–priority token optimization (DPTO) strategy, achieving higher attack success rates and improved efficiency across open- and closed-source LLMs.

---

## Installation

Clone the repository and install the dependencies:

```bash
cd TAO-Attack
pip install -r requirements.txt
```

You may also install it as a package:

```bash
pip install .
```

---

## Quick Start

Run the main attack implementation:

```bash
python attack.py
```


---

## Windows `.exe` Packaging (PyInstaller)

You can build standalone Windows executables for both CLI and GUI entrypoints.

### Local build (Windows)

```bash
pip install -r requirements.txt
pyinstaller --noconfirm attack.spec
pyinstaller --noconfirm gui.spec
```

Artifacts are generated in `dist/`:

- `dist/attack.exe` (CLI)
- `dist/gui.exe` (GUI launcher)

### Run

- CLI: `dist/attack.exe --help`
- GUI: start `dist/gui.exe` and open `http://127.0.0.1:7860`

> `gui.exe` expects `attack.exe` to be in the same `dist/` folder so it can launch attacks from the GUI.

### CI build

A Windows GitHub Actions workflow is included at:

- `.github/workflows/windows-exe-build.yml`

It builds both executables, runs a smoke test, and uploads artifacts.

### Limitations

- Large model weights are **not** bundled into the executable.
- Keep model files external and pass local paths (or HuggingFace model IDs) at runtime.

---

## Project Structure

```
LLM-ATTACKS/
├── api_experiments/       
├── data/                  # Data resources (e.g., evaluation prompts)
├── experiments/           
├── llm_attacks/           
├── attack.py              # Implementation of TAO-Attack
├── check_openai.py        # Helper script for evaluation with OpenAI APIs
├── requirements.txt       # Python dependencies
├── setup.py               # Installation script
├── LICENSE                # License file (MIT)
└── README.md              # Project description (this file)
```

---

## GUI

TAO-Attack includes a Gradio-based web GUI for configuring, running, and inspecting attacks.

### Launch the GUI

```bash
python gui.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Features

| Tab | Description |
|---|---|
| **⚙️ Configuration & Launch** | Set model path, dataset path, save folder, and hyperparameters. Includes fast/balanced/thorough presets, dry-run validation, resume mode, and safe-mode confirmation gate. |
| **📋 Attack Data** | Browse and filter behaviors/targets with category, index-range, and keyword filters. |
| **📜 Live Logs** | Stream stdout from `attack.py` and monitor runtime, ETA, and GPU memory usage. |
| **📊 Results** | Select and view JSONL result files with per-behaviour success/failure summaries. |
| **📈 Run History & Export** | Compare runs (success rate, avg queries, avg loss) and export CSV/JSON/SVG artifacts. |

### CLI options used by the GUI

The GUI now launches `attack.py` with:

- `--model_path`
- `--save_folder`
- `--data_path`
- `--cl_threshold`
- `--num_steps`
- `--batch_size`
- `--topk`
- `--temp`
- `--alpha`
- `--beta`
- `--start_bidx` (when resume mode is enabled)
- `--config_path` (GUI-internal launch config file)

---

## License

This project is licensed under the terms of the MIT License.
See the [LICENSE](LICENSE) file for details.
