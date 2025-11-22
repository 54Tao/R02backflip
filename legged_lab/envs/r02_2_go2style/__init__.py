"""RO2 Go2-style backflip training module"""

from .config import (
    R02Go2StyleEnvCfg,
    R02Go2StyleRewardCfg,
    R02Go2StyleTerminationsCfg,
    R02Go2StyleFlatEnvCfg,
    R02Go2StyleAgentCfg,
    R02Go2StyleFlatAgentCfg,
)

from .go2style_env import Go2StyleEnv

from . import rewards

__all__ = [
    "R02Go2StyleEnvCfg",
    "R02Go2StyleRewardCfg",
    "R02Go2StyleTerminationsCfg",
    "R02Go2StyleFlatEnvCfg",
    "R02Go2StyleAgentCfg",
    "R02Go2StyleFlatAgentCfg",
    "Go2StyleEnv",
    "rewards",
]
