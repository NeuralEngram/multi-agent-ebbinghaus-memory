""" 
    Implements Ebbinghaus Forgetting Curve for memory retention scoring in 
    a multi-agent system.

    Using the retention formula: 
    R=e^(-t/S) 
    where, 
        R = retention strength, lies between 0.0 to 1.0
        t = time elapsed since last review (in hours)
        S = stability factor (increases with each successful recall) 
"""

import math 
from datetime import datetime, timezone
from typing import Optional


#Memory decay configuration
RETENTION_THRESHOLD = 0.3         #Memory is considered "Forgotten" below this value
REINFORCEMENT_BOOST = 1.8         #Stability multiplier after a recall
BASE_STABLE_HOURS = 24.0          #Default stability for new memories
MAX_STABLE_HOURS = 8760.0         #Maximum stability of 1 Year


def compute_retention(
    last_reviewed_at: datetime,
    stability_hours: float,
    now: Optional[datetime] = None, ) -> float:

    """
    Calculate the current retention strength of a memory item.
    Args:
        last_reviewed_at: UTC datetime when the memory was last accessed.
        stability_hours:  Current stability value (S) for this memory.
        now:              Reference time (defaults to utc now).

    Returns:
        Retention score in [0.0, 1.0].
    """

    if stability_hours <= 0:
        raise ValueError("stability_hours must be positive")

    if last_reviewed_at.tzinfo is None:
        raise ValueError("last_reviewed_at must be timezone-aware")

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    
    stability_hours = min(stability_hours, MAX_STABLE_HOURS)

    elapsed_hours = max((now - last_reviewed_at).total_seconds() / 3600.0, 0.0)
    retention = math.exp(-elapsed_hours / stability_hours)

    return max(0.0, min(round(retention, 6), 1.0))

