"""
Traffic generation scenarios for the QEMU testbed.
These scripts run inside the client VM to generate realistic traffic.
"""

import subprocess
import time
from typing import Optional


class TrafficScenario:
    """Base class for traffic generation scenarios."""
    
    def __init__(self, server_ip: str = "192.168.100.20", duration: int = 10):
        self.server_ip = server_ip
        self.duration = duration
    
    def run(self) -> None:
        raise NotImplementedError


class SSHFileOperations(TrafficScenario):
    """
    SSH scenario: create large random files, perform operations,
    execute commands, modify permissions, and cleanup.
    As used in the TATA paper.
    """
    
    def __init__(self, server_ip: str = "192.168.100.20", duration: int = 10, ssh_user: str = "tata"):
        super().__init__(server_ip, duration)
        self.ssh_user = ssh_user
    
    def run(self) -> None:
        """Run SSH file operations against the server."""
        ssh_target = f"{self.ssh_user}@{self.server_ip}"
        
        commands = [
            # Create large random files
            f"ssh {ssh_target} 'dd if=/dev/urandom of=/tmp/tata_large_1.bin bs=1M count=10'",
            f"ssh {ssh_target} 'dd if=/dev/urandom of=/tmp/tata_large_2.bin bs=1M count=5'",
            # Frequent file operations
            f"ssh {ssh_target} 'for i in $(seq 1 50); do echo $i > /tmp/tata_file_$i.txt; done'",
            f"ssh {ssh_target} 'for f in /tmp/tata_file_*.txt; do cat $f >> /tmp/tata_combined.txt; done'",
            # Complex commands
            f"ssh {ssh_target} 'ps aux | grep ssh | wc -l'",
            f"ssh {ssh_target} 'find /tmp -name \"tata_*\" | sort | head -20'",
            # Modify permissions
            f"ssh {ssh_target} 'chmod 600 /tmp/tata_large_1.bin'",
            f"ssh {ssh_target} 'chmod 644 /tmp/tata_large_2.bin'",
            # Cleanup
            f"ssh {ssh_target} 'rm -f /tmp/tata_*'",
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, timeout=30, capture_output=True)
            except subprocess.TimeoutExpired:
                pass


class HTTPTraffic(TrafficScenario):
    """HTTP scenario: make various web requests to the server."""
    
    def run(self) -> None:
        urls = [
            f"http://{self.server_ip}/",
            f"http://{self.server_ip}/nonexistent",
            f"http://{self.server_ip}/index.html",
        ]
        
        for url in urls:
            try:
                subprocess.run(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                    timeout=10,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired:
                pass


class DNSTraffic(TrafficScenario):
    """DNS scenario: perform DNS queries."""
    
    def run(self) -> None:
        domains = [
            "google.com",
            "cloudflare.com",
            "github.com",
            "example.com",
        ]
        
        for domain in domains:
            try:
                subprocess.run(
                    ["dig", "+short", domain],
                    timeout=5,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired:
                pass


class IPerf3Traffic(TrafficScenario):
    """iPerf3 scenario: measure bandwidth between client and server."""
    
    def __init__(self, server_ip: str = "192.168.100.20", duration: int = 10, port: int = 5201):
        super().__init__(server_ip, duration)
        self.port = port
    
    def run(self) -> None:
        try:
            # TCP test
            subprocess.run(
                ["iperf3", "-c", self.server_ip, "-p", str(self.port), "-t", str(self.duration)],
                timeout=self.duration + 10,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            pass


SCENARIO_MAP = {
    "ssh": SSHFileOperations,
    "http": HTTPTraffic,
    "dns": DNSTraffic,
    "iperf3": IPerf3Traffic,
}


def run_scenario(name: str, server_ip: str = "192.168.100.20", duration: int = 10) -> None:
    """Run a traffic scenario by name."""
    if name not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_MAP.keys())}")
    
    scenario = SCENARIO_MAP[name](server_ip=server_ip, duration=duration)
    scenario.run()
