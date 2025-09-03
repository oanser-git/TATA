"""
Abstract base class for network testbed.
Phase 2: used by RL environment to generate real traffic.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class NetworkTestbed(ABC):
    """
    Abstract interface for a configurable traffic testbed.
    Implementations: QEMU/KVM testbed, Docker testbed, etc.
    """
    
    @abstractmethod
    def apply_config(self, action: np.ndarray) -> None:
        """
        Apply traffic-shaping configuration.
        
        Args:
            action: Continuous vector representing:
                [loss%, jitter_ms, delay_ms, duplication%, corruption%, reordering%, correlation%]
        """
        pass
    
    @abstractmethod
    def generate_traffic(self, scenario: str = "ssh") -> str:
        """
        Execute traffic generation scenario.
        
        Args:
            scenario: Type of traffic to generate (ssh, http, etc.).
        
        Returns:
            Path to captured .pcap file.
        """
        pass
    
    @abstractmethod
    def collect_flows(self, pcap_path: str) -> np.ndarray:
        """
        Convert pcap to flow features using CICFlowMeter.
        
        Args:
            pcap_path: Path to pcap file.
        
        Returns:
            Flow feature vector (1, n_features).
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset testbed to default state."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Teardown testbed resources."""
        pass
