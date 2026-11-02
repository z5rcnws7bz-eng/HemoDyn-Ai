# Contributing to HemoDyn-AI

We welcome contributions to the HemoDyn-AI research framework! Whether you are interested in extending geometries, integrating pulsatile waveform boundary conditions, testing alternative scientific machine learning architectures (e.g., PINNs, Neural Operators, Graph Neural Networks), or improving computational performance, your help is appreciated.

---

## Code of Conduct

All contributors are expected to uphold an inclusive, respectful, and scientifically rigorous research environment.

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/z5rcnws7bz-eng/HemoDyn-Ai.git
   cd HemoDyn-Ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify the installation:**
   ```bash
   pytest tests/ -v
   ```

---

## Guidelines for Research Contributions

1. **Preserve Determinism & Reproducibility:**
   - Set fixed random seeds (`random_state=42`) for all machine learning pipelines.
   - Do not commit large binary checkpoints (`*.bin` files) or intermediate restart directories to git.
   - Document any changes to physical parameters ($\mu$, $\rho$, $Q$) in `REPRODUCIBILITY.md`.

2. **Testing:**
   - Add automated test assertions in `tests/test_reproducibility.py` for any new geometry or model formulation.
   - Ensure all existing tests pass before submitting a pull request.

3. **Pull Request Protocol:**
   - Use descriptive commit messages.
   - Summarize the scientific rationale and benchmark impact in your PR description.

