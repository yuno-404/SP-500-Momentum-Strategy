"""
Data Cache Manager - Avoid redundant downloads
"""

import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class CacheManager:
    """Manage cached data to avoid redundant downloads"""

    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, data_type, **kwargs):
        """Generate cache file path"""
        # Create unique filename based on parameters
        if data_type == "stock_prices":
            start_date = kwargs.get("start_date", "unknown")
            return self.cache_dir / f"stock_prices_{start_date}.pkl"

        elif data_type == "sector_weights":
            start_date = kwargs.get("start_date", "unknown")
            return self.cache_dir / f"sector_weights_{start_date}.pkl"

        elif data_type == "components":
            return self.cache_dir / "sp500_components.pkl"

        elif data_type == "benchmark":
            start_date = kwargs.get("start_date", "unknown")
            return self.cache_dir / f"benchmark_{start_date}.pkl"

        return None

    def is_cache_valid(self, cache_path, max_age_hours=24):
        """Check if cache exists and is not too old"""
        if not cache_path.exists():
            return False

        # Check file age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time

        return age < timedelta(hours=max_age_hours)

    def load_cache(self, data_type, **kwargs):
        """Load data from cache if valid"""
        cache_path = self.get_cache_path(data_type, **kwargs)

        if cache_path and self.is_cache_valid(cache_path):
            print(f"[INFO] Loading {data_type} from cache...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        return None

    def save_cache(self, data, data_type, **kwargs):
        """Save data to cache"""
        cache_path = self.get_cache_path(data_type, **kwargs)

        if cache_path:
            print(f"[INFO] Saving {data_type} to cache...")
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

    def clear_cache(self, data_type=None):
        """Clear cache files"""
        if data_type:
            cache_path = self.get_cache_path(data_type)
            if cache_path and cache_path.exists():
                cache_path.unlink()
                print(f"[INFO] Cleared {data_type} cache")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            print("[INFO] Cleared all cache")

    def get_cache_info(self):
        """Get information about cached data"""
        info = []

        for file in self.cache_dir.glob("*.pkl"):
            file_time = datetime.fromtimestamp(file.stat().st_mtime)
            size_mb = file.stat().st_size / (1024 * 1024)
            age = datetime.now() - file_time

            info.append(
                {
                    "name": file.name,
                    "size_mb": f"{size_mb:.2f}",
                    "age_hours": f"{age.total_seconds() / 3600:.1f}",
                    "timestamp": file_time.strftime("%Y-%m-%d %H:%M"),
                }
            )

        return info


if __name__ == "__main__":
    # Test cache manager
    cache = CacheManager()

    # Test save
    test_data = pd.DataFrame({"A": [1, 2, 3]})
    cache.save_cache(test_data, "stock_prices", start_date="2020-01-01")

    # Test load
    loaded = cache.load_cache("stock_prices", start_date="2020-01-01")
    print(f"\nLoaded data:\n{loaded}")

    # Test info
    info = cache.get_cache_info()
    print(f"\nCache info:")
    for item in info:
        print(f"  {item}")

    # Test clear
    cache.clear_cache()
