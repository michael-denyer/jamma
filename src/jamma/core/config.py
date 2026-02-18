"""Configuration dataclasses for JAMMA.

This module contains dataclasses that configure various aspects of JAMMA
execution, including output paths and logging settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OutputConfig:
    """Configuration for output files and directories.

    Attributes:
        outdir: Output directory for result files. Created if it doesn't exist.
        prefix: Prefix for output filenames (e.g., "result" produces "result.log.txt").
        verbose: Enable verbose/debug output to console.
    """

    outdir: Path = field(default_factory=lambda: Path("output"))
    prefix: str = "result"
    verbose: bool = False

    def __post_init__(self) -> None:
        if os.sep in self.prefix or "/" in self.prefix:
            raise ValueError(
                f"prefix must not contain path separators, "
                f"got '{self.prefix}'. Use outdir for directory paths."
            )

    @property
    def log_path(self) -> Path:
        """Path to the GEMMA-compatible log file.

        Returns:
            Path to {outdir}/{prefix}.log.txt
        """
        return self.outdir / f"{self.prefix}.log.txt"

    def ensure_outdir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.outdir.mkdir(parents=True, exist_ok=True)
