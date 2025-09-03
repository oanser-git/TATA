"""
QEMU/KVM testbed implementation.
Controls two VMs (client + server) via SSH to generate real network traffic.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import paramiko  # type: ignore[import]

from tata.testbed.base import NetworkTestbed
from tata.testbed.flow_converter import CICFlowMeterConverter
from tata.testbed.scenarios import run_scenario


class QemuTestbed(NetworkTestbed):
    """
    QEMU/KVM testbed for realistic traffic generation.
    
    Architecture:
      - Host with KVM support
      - Two QEMU VMs: Client (192.168.100.10) and Server (192.168.100.20)
      - Both VMs connected via Linux bridge (br-tata)
      - Client applies tc qdisc (netem) on its egress NIC
      - tcpdump captures on bridge
      - CICFlowMeter converts pcap to CSV
    
    Requirements:
      - KVM enabled host
      - VM images at configured paths
      - cloud-init ISOs for auto-configuration
      - root access for bridge/tap creation
      - CICFlowMeter Java jar installed
    
    Usage:
      1. Build VM images (see scripts/qemu/README.md)
      2. Run `sudo scripts/qemu/testbed.sh start`
      3. Instantiate QemuTestbed
      4. Call apply_config(), generate_traffic(), collect_flows()
      5. Run `sudo scripts/qemu/testbed.sh stop` when done
    """
    
    def __init__(
        self,
        client_ip: str = "192.168.100.10",
        server_ip: str = "192.168.100.20",
        ssh_user: str = "tata",
        ssh_key_path: str = "scripts/qemu/cloud-init/tata_vm_key",
        capture_dir: str = "/tmp/tata_captures",
        cicflowmeter_jar: str = "tools/CICFlowMeter-4.0/bin/CICFlowMeter",
        bridge_name: str = "br-tata",
    ):
        self.client_ip = client_ip
        self.server_ip = server_ip
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.bridge_name = bridge_name
        
        self.cicflowmeter = CICFlowMeterConverter(cicflowmeter_jar)
        self.current_config = None
        self.ssh_client = None
        
        # Verify prerequisites
        if not Path(ssh_key_path).exists():
            raise FileNotFoundError(f"SSH key not found: {ssh_key_path}")
    
    def _get_ssh_client(self) -> "paramiko.SSHClient":
        """Create and return an SSH client to the client VM."""
        if self.ssh_client is None:
            client = paramiko.SSHClient()  # type: ignore[misc]
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # type: ignore[misc]
            client.connect(
                hostname=self.client_ip,
                username=self.ssh_user,
                key_filename=self.ssh_key_path,
                timeout=30,
            )
            self.ssh_client = client
        return self.ssh_client
    
    def apply_config(self, action: np.ndarray) -> None:
        """
        Apply tc netem configuration on client VM.
        Maps normalized action [0,1] to actual traffic shaping parameters.
        """
        action = np.clip(action, 0.0, 1.0)
        
        # Map [0,1] to paper's parameter ranges
        loss = float(np.clip(action[0] * 10, 5.0, 10.0))       # 5% - 10%
        jitter = float(np.clip(action[1] * 10, 4.0, 10.0))     # 4ms - 10ms
        delay = float(np.clip(action[2] * 40, 10.0, 40.0))     # 10ms - 40ms
        duplication = float(np.clip(action[3] * 5, 0.1, 5.0))   # 0.1% - 5%
        corruption = float(np.clip(action[4] * 10, 0.1, 10.0))  # 0.1% - 10%
        reordering = float(np.clip(action[5] * 50, 0.1, 50.0))  # 0.1% - 50%
        correlation = float(np.clip(action[6] * 100, 50.0, 100.0))  # 50% - 100%
        
        self.current_config = {
            "loss": loss, "jitter": jitter, "delay": delay,
            "duplication": duplication, "corruption": corruption,
            "reordering": reordering, "correlation": correlation,
        }
        
        # Build tc netem command
        # Apply on eth0 (or whatever the main interface is)
        tc_cmd = (
            f"sudo tc qdisc del dev eth0 root 2>/dev/null; "
            f"sudo tc qdisc add dev eth0 root netem "
            f"delay {delay}ms {jitter}ms {correlation}% "
            f"loss {loss}% "
            f"duplicate {duplication}% "
            f"corrupt {corruption}% "
            f"reorder {reordering}%"
        )
        
        ssh = self._get_ssh_client()
        stdin, stdout, stderr = ssh.exec_command(tc_cmd)
        exit_code = stdout.channel.recv_exit_status()
        
        if exit_code != 0:
            err = stderr.read().decode().strip()
            raise RuntimeError(f"tc qdisc failed (exit={exit_code}): {err}")
    
    def generate_traffic(self, scenario: str = "ssh") -> str:
        """
        Generate traffic between client and server VMs.
        
        Args:
            scenario: Type of traffic (ssh, http, dns, iperf3).
        
        Returns:
            Path to captured .pcap file.
        """
        pcap_path = self.capture_dir / f"capture_{int(time.time())}.pcap"
        
        # Start tcpdump on host bridge (background)
        tcpdump_cmd = [
            "sudo", "tcpdump", "-i", self.bridge_name,
            "-w", str(pcap_path),
            "-s", "0",  # Capture full packets
            "-n",       # Don't resolve hostnames
        ]
        
        tcpdump_proc = subprocess.Popen(
            tcpdump_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Give tcpdump time to start
        time.sleep(2)
        
        try:
            # Run traffic scenario on client VM via SSH
            ssh = self._get_ssh_client()
            scenario_script = (
                f"python3 -c \""
                f"from tata.testbed.scenarios import run_scenario; "
                f"run_scenario('{scenario}', '{self.server_ip}', 10)"
                f"\""
            )
            stdin, stdout, stderr = ssh.exec_command(scenario_script)
            exit_code = stdout.channel.recv_exit_status()
            
            # Wait a bit for flows to complete
            time.sleep(3)
            
        finally:
            # Stop tcpdump
            tcpdump_proc.terminate()
            try:
                tcpdump_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tcpdump_proc.kill()
        
        return str(pcap_path)
    
    def collect_flows(self, pcap_path: str) -> np.ndarray:
        """
        Convert pcap to flows using CICFlowMeter.
        
        Args:
            pcap_path: Path to pcap file.
        
        Returns:
            Flow feature vector (1, n_features).
        """
        features = self.cicflowmeter.extract_first_flow(pcap_path)
        if features is None:
            raise RuntimeError("No flows extracted from pcap")
        return features.reshape(1, -1)
    
    def reset(self) -> None:
        """Reset tc rules to default (no shaping)."""
        if self.ssh_client:
            tc_cmd = "sudo tc qdisc del dev eth0 root 2>/dev/null || true"
            self.ssh_client.exec_command(tc_cmd)
        self.current_config = None
    
    def close(self) -> None:
        """Close SSH connection."""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
