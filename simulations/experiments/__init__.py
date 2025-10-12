"""
Intent-aware planning experiments package
"""
from .config import (
    ExperimentConfig, 
    get_config_mppi_particle,
    get_config_mppi_worst,
    get_config_probabilistic,
    get_config_custom,
    # Aliases for backward compatibility
    get_config_predict_tv,
    get_config_worst_case
)
from .runner import IntentAwarePlanner, run_experiment

__all__ = [
    'ExperimentConfig',
    'IntentAwarePlanner',
    'run_experiment',
    'get_config_mppi_particle',
    'get_config_mppi_worst',
    'get_config_probabilistic',
    'get_config_custom',
    'get_config_predict_tv',
    'get_config_worst_case'
]