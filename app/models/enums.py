import enum


class ProcessingStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OverallStatus(str, enum.Enum):
    """High-level verdict once all analyzers have run."""
    OK = "OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"
