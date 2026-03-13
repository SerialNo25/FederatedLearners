from pathlib import Path
from typing import Protocol


class Stage(Protocol):
    def execute(self) -> Path:
        ...