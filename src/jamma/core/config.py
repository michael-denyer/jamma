"""Configuration dataclasses for GEMMA-Next.

This module contains dataclasses that configure various aspects of GEMMA-Next
execution, including output paths and logging settings.
"""

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
