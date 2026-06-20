"""A single simulated vehicle on an approach lane.

Distance is measured as `d` = metres to the stop line: large positive values are
far upstream, 0 is the stop line, negative values mean the vehicle has entered /
crossed the intersection. Vehicles are removed once they clear the intersection box.
"""

from dataclasses import dataclass, field
from itertools import count

_ids = count()


@dataclass
class Vehicle:
    approach: str            # 'N' | 'E' | 'S' | 'W'
    lane: int                # lane index on that approach
    d: float                 # metres to stop line (decreasing as it advances)
    spawn_time: float        # sim time the vehicle entered the network
    speed: float = 0.0       # m/s
    emergency: bool = False  # higher-weight in the objective (hook for M3)
    delay: float = 0.0       # accumulated delay (vehicle-seconds below free flow)
    wait: float = 0.0        # accumulated time effectively stopped (for fairness cap)
    release: float = 0.0     # how long the path ahead has been clear (start-up reaction)
    vid: int = field(default_factory=lambda: next(_ids))

    @property
    def crossed(self) -> bool:
        """True once the front of the vehicle is past the stop line."""
        return self.d <= 0.0
