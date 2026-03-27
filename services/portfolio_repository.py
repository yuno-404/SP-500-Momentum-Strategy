import json
from pathlib import Path


class PortfolioRepository:
    """Persist portfolio state as JSON."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def empty_state(self):
        return {
            "holdings": [],
            "history": [],
            "initial_capital": 0,
            "cash": 0,
            "start_date": None,
        }

    def load(self):
        if self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    base = self.empty_state()
                    base.update(data)
                    return base
        return self.empty_state()

    def save(self, state: dict):
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    def reset(self):
        if self.file_path.exists():
            self.file_path.unlink()
