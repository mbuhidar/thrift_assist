"""
Memory monitoring and optimization utilities.
"""

import psutil
import os
import gc
import logging

logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(context: str = ""):
    """Log current memory usage."""
    memory_mb = get_memory_usage_mb()
    logger.info(f"💾 Memory usage {context}: {memory_mb:.2f} MB")
    return memory_mb


def force_garbage_collection():
    """Force aggressive garbage collection."""
    gc.collect()
    gc.collect()  # Run twice for more thorough cleanup
    gc.collect()


def check_memory_limit(limit_mb: float = 512) -> bool:
    """Check if memory usage exceeds limit."""
    current = get_memory_usage_mb()
    if current > limit_mb:
        logger.warning(f"⚠️ Memory usage ({current:.2f}MB) exceeds limit ({limit_mb}MB)")
        force_garbage_collection()
        return False
    return True


def get_memory_info() -> dict:
    """Get detailed memory information."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "rss_mb": mem_info.rss / 1024 / 1024,
        "vms_mb": mem_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024,
    }


def log_memory_info(context: str = ""):
    """Log detailed memory information."""
    info = get_memory_info()
    logger.info(
        f"💾 Memory {context}: "
        f"RSS={info['rss_mb']:.1f}MB "
        f"VMS={info['vms_mb']:.1f}MB "
        f"({info['percent']:.1f}% of system)"
    )
    return info
