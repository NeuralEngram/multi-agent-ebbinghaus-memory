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
from typing import Optional, List, Tuple


#Memory decay configuration
RETENTION_THRESHOLD = 0.3         #Memory is considered "Forgotten" below this value
DEFAULT_REINFORCEMENT_BOOST = 1.8
DEFAULT_DIMINISHING_FACTOR = 0.3
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

    return max(0.0, min(retention,1.0))


def reinforce_memory(
    stability_hours: float,
    quality: float = 1.0,
    review_count: int = 0,
    boost: float = DEFAULT_REINFORCEMENT_BOOST,
    diminishing_factor: float = DEFAULT_DIMINISHING_FACTOR
) -> float:

    """
        After a successful recall, stability increases.
        Each reinforcement multiplies the stability factor.
        The more times a memory is reviewed, the longer it persists.

        Args:
            stability_hours: Current stability before reinforcement.
            quality: Recall quality from 0.0 (weak) to 1.0 (perfect).
             Weak recalls give a smaller stability boost.
        
        Returns:
            New stability value, capped at MAX_STABLE_HOURS.
    """
    if stability_hours <= 0:
        raise ValueError("stability_hours must be positive")
    
    if not 0.0 <=quality <= 1.0:
        raise ValueError("quality must be between 0 and 1")
    
    if review_count < 0:
        raise ValueError("review_count must be >= 0")
    
    decay = 1 / (1 + diminishing_factor * math.log1p(review_count))
    effective_boost = 1.0 + quality * (boost - 1.0) * decay
    new_stability = stability_hours * effective_boost
    return min(new_stability, MAX_STABLE_HOURS)


def is_memory_forgotten(retention: float) -> bool:

    """
        Return True if retention has dropped below the forgetting threshold.
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must be between 0 and 1")

    return retention < RETENTION_THRESHOLD


def time_until_forgotten(
    last_reviewed_at: datetime,
    stability_hours: float,
    now: Optional[datetime] = None, ) -> float:

    """
    Estimate how many hours remain before this memory crosses the forgetting threshold.

    Returns:
        Hours remaining (0.0 if already forgotten).
    """
    if stability_hours <= 0:
        raise ValueError("stability_hours must be positive")
    
    if last_reviewed_at.tzinfo is None:
        raise ValueError("last_reviewed_at must be timezone-aware")
    
    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")


    elapsed_hours = max((now - last_reviewed_at).total_seconds() / 3600.0, 0.0)

    # Solve: RETENTION_THRESHOLD = e^(-(elapsed + remaining) / S)
    # → remaining = -S * ln(threshold) - elapsed
    stability_hours = min(stability_hours, MAX_STABLE_HOURS)
    threshold_hours = -stability_hours * math.log(RETENTION_THRESHOLD)
    remaining = threshold_hours - elapsed_hours
    return max(remaining, 0.0)


def decay_curve_points(
    stability_hours: float,
    num_points: int = 20, ) -> List[Tuple[float, float]]:

    """
    Generate (time_hours, retention) data points along the decay curve.
    Useful for debugging or visualisation.

    Args:
        stability_hours: Stability S to model.
        num_points:      How many sample points to generate.

    Returns:
        List of (elapsed_hours, retention) tuples.
    """
    if stability_hours <= 0:
        raise ValueError("stability_hours must be positive")

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    
    stability_hours = min(stability_hours, MAX_STABLE_HOURS)

    # Sample up to 3× the point where retention hits the threshold
    t_max = -stability_hours * math.log(RETENTION_THRESHOLD) * 3
    step = t_max / max(num_points - 1, 1)

    return [
       (i * step, math.exp(-(i * step) / stability_hours))
        for i in range(num_points)
    ]