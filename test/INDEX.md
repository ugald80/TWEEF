# Tests and experiments

Keep all test cases, experimental definitions, runners, datasets, reports and
generated results in this directory. The repository README documents the library
and links here; experimental findings belong in the experiment report.

| Location | Contents |
|---|---|
| `test_*.py` | Automated unit, regression and integration tests |
| `datasets/` | German, COMPAS and Adult input data |
| `experiments/` | Experimental runners, executable examples and requirements |
| [experiments/REPORT.md](experiments/REPORT.md) | Protocols, results, figures and interpretation |
| `experiments/results/` | Saved metrics, manifests, caches, logs and audits |

Run commands from the repository root.

```sh
python -m pytest -q
TWEEF_RUN_INTEGRATION=1 python -m pytest -q
```

The complete integration suite needs the optional fairness and interpretability
dependencies. The pinned German environment is defined in
[`experiments/requirements-german.txt`](experiments/requirements-german.txt).

```sh
.venv-paper/bin/python test/experiments/example_final_surrogate.py
.venv-paper/bin/python test/experiments/german_final_surrogate.py --help
```

Archived runs retain their original manifests, hashes and logs, including the
paths valid when they were executed. To audit the saved final-surrogate run after
this directory relocation, use the explicit archived-source mode:

```sh
.venv-paper/bin/python test/experiments/verify_final_surrogate.py \
  --archived-source --verify
```

This preserves the original `verification.json` and writes a separate relocation
audit. New executions use the relocated runners and their own source fingerprints;
use a fresh output directory rather than overwriting archived runs:

```sh
.venv-paper/bin/python test/experiments/german_final_surrogate.py \
  --output test/experiments/results/german_final_surrogate_new
```
