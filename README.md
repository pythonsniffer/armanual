# armanual — Bimanual VLA Manipulation with Multi-Modal Reasoning

Dinner-table setup with **two simulated SO-101 arms in MuJoCo**, driven by natural-language
(text or speech) instructions, closed-loop camera observation, and a VLA policy — with inference
optimized via **OpenVINO** for **Intel Core Ultra Series 2/3**.

Submission for the Intel Physical AI Online Challenge.

> **Status: under active development.** This README tracks what is actually implemented and
> verified. Claims here are backed by a command you can run; anything unverified is listed in
> [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m armanual.cli verify      # environment smoke test
```

## Documentation

| Doc | Purpose |
| --- | --- |
| [docs/REPO_RECON.md](docs/REPO_RECON.md) | Phase 0 reconnaissance: environment, assets, tooling |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Requirement traceability (official PS / user / engineering / open) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and data flow |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Known limitations and unverified claims |

## License

Code: Apache-2.0. Vendored third-party assets keep their own licenses — see
[assets/ATTRIBUTION.md](assets/ATTRIBUTION.md).
