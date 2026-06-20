"""Signal controllers: engineered baselines + the learning controller."""

from .actuated import Actuated
from .base import Controller
from .dqn import DQN
from .fixed_time import FixedTime
from .max_pressure import MaxPressure

BASELINES = {
    "fixed_time": FixedTime,
    "actuated": Actuated,
    "max_pressure": MaxPressure,
}

__all__ = ["Controller", "FixedTime", "Actuated", "MaxPressure", "DQN", "BASELINES"]
