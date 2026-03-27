"""Environment variable helpers."""

from dotenv import load_dotenv
import os


load_dotenv()


def get_fmp_api_key(required=False):
    """Return FMP API key from environment (.env supported)."""
    key = os.getenv("FMP_API_KEY")
    if required and not key:
        raise RuntimeError(
            "Missing FMP_API_KEY. Add it to .env or system environment variables."
        )
    return key
