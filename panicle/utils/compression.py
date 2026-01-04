"""
Compression utilities with optional parallel gzip (pigz) support.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union
import pandas as pd


def _pigz_available() -> bool:
    """Check if pigz is available and working on the system."""
    if shutil.which('pigz') is None:
        return False
    # Verify pigz actually works
    try:
        result = subprocess.run(
            ['pigz', '--version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# Cache the result at module load time
PIGZ_AVAILABLE = _pigz_available()


def to_csv_gzip(df: pd.DataFrame,
                filepath: Union[str, Path],
                **csv_kwargs) -> None:
    """
    Write DataFrame to gzip-compressed CSV, using pigz for parallel compression if available.

    Falls back to standard gzip if pigz is not installed or fails.

    Args:
        df: DataFrame to write
        filepath: Output path (should end in .gz)
        **csv_kwargs: Additional arguments passed to DataFrame.to_csv()

    Note:
        Install pigz for faster parallel compression:
        - Ubuntu/Debian: sudo apt install pigz
        - macOS: brew install pigz
        - conda: conda install pigz
    """
    filepath = Path(filepath)

    if PIGZ_AVAILABLE:
        # Use pigz for parallel compression
        # Write uncompressed CSV to temp file, then compress with pigz
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp, **csv_kwargs)

        try:
            # pigz uses all available cores by default
            with open(filepath, 'wb') as out_file:
                subprocess.run(
                    ['pigz', '-c', tmp_path],
                    stdout=out_file,
                    check=True
                )
        except subprocess.CalledProcessError:
            # If pigz fails, fall back to standard gzip
            Path(tmp_path).unlink(missing_ok=True)
            df.to_csv(filepath, compression='gzip', **csv_kwargs)
            return
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    else:
        # Fall back to standard gzip
        df.to_csv(filepath, compression='gzip', **csv_kwargs)


def get_compression_info() -> str:
    """Return info about available compression methods."""
    if PIGZ_AVAILABLE:
        try:
            result = subprocess.run(['pigz', '--version'], capture_output=True, text=True)
            version = result.stdout.strip() or result.stderr.strip()
            return f"pigz (parallel gzip): {version}"
        except Exception:
            return "pigz (parallel gzip): available"
    else:
        return "gzip (single-threaded) - install pigz for faster parallel compression"
