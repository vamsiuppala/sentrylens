"""
Structured logging configuration using loguru.
Provides consistent logging across the application.
"""
import sys
import json
from pathlib import Path
from loguru import logger
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from configs.settings import settings


class InterceptHandler:
    """Intercept standard logging and redirect to loguru."""
    
    def write(self, message: str) -> None:
        if message.strip():
            logger.opt(depth=6, exception=None).info(message.strip())
    
    def flush(self) -> None:
        pass


def serialize_record(record: Dict[str, Any]) -> str:
    """Serialize log record to JSON format."""
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra fields if present
    if record.get("extra"):
        subset["extra"] = record["extra"]
    
    # Add exception if present
    if record.get("exception"):
        subset["exception"] = str(record["exception"])
    
    return json.dumps(subset)


def setup_logger() -> None:
    """Configure loguru logger with structured output."""
    
    # Remove default handler
    logger.remove()
    
    # Add handler based on format preference
    if settings.LOG_FORMAT == "json":
        logger.add(
            sys.stderr,
            level=settings.LOG_LEVEL,
            serialize=lambda record: serialize_record(record) + "\n",
        )
    else:
        logger.add(
            sys.stderr,
            level=settings.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
    
    # Also log to file
    log_dir = settings.PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    if settings.LOG_FORMAT == "json":
        logger.add(
            log_dir / "sentrylens_{time}.log",
            rotation="100 MB",
            retention="10 days",
            level=settings.LOG_LEVEL,
            serialize=lambda record: serialize_record(record) + "\n",
        )
    else:
        logger.add(
            log_dir / "sentrylens_{time}.log",
            rotation="100 MB",
            retention="10 days",
            level=settings.LOG_LEVEL,
        )
    
    logger.info("Logger initialized", log_level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)


# Initialize logger on import
setup_logger()

# Export configured logger
__all__ = ["logger"]