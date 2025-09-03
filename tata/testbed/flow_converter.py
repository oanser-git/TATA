"""
CICFlowMeter wrapper for converting pcap files to flow CSVs.
Uses the GintsEngelen/CICFlowMeter Java implementation.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class CICFlowMeterConverter:
    """
    Wrapper for CICFlowMeter Java application.
    Converts pcap files to bidirectional flow CSVs.
    """
    
    def __init__(self, jar_path: str = "tools/CICFlowMeter-4.0/bin/CICFlowMeter"):
        """
        Args:
            jar_path: Path to CICFlowMeter executable or jar.
                      Can be the Java jar path or a wrapper script.
        """
        self.jar_path = Path(jar_path)
        if not self.jar_path.exists():
            # Try common locations
            alt_paths = [
                "CICFlowMeter",
                "/usr/local/bin/CICFlowMeter",
                "/opt/CICFlowMeter/bin/CICFlowMeter",
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    self.jar_path = Path(alt)
                    break
    
    def convert(
        self,
        pcap_path: str,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Convert pcap to flows.
        
        Args:
            pcap_path: Path to pcap file.
            output_dir: Directory to save CSV. If None, uses temp dir.
        
        Returns:
            DataFrame with flow features.
        """
        pcap = Path(pcap_path)
        if not pcap.exists():
            raise FileNotFoundError(f"Pcap not found: {pcap_path}")
        
        if output_dir is None:
            out = Path(tempfile.mkdtemp(prefix="tata_flows_"))
        else:
            out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # Run CICFlowMeter
        # It typically outputs to output_dir/ and creates a CSV with the same name as pcap
        cmd = [
            "java", "-jar", str(self.jar_path),
            str(pcap_path),
            str(output_dir),
        ]
        
        if str(self.jar_path).endswith("CICFlowMeter") and not str(self.jar_path).endswith(".jar"):
            # It might be a shell wrapper script
            cmd = [str(self.jar_path), str(pcap_path), str(output_dir)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("CICFlowMeter conversion timed out")
        
        # Find generated CSV
        csv_files = list(out.glob("*.csv"))
        if not csv_files:
            # Sometimes CICFlowMeter creates subdirectories
            csv_files = list(out.rglob("*.csv"))
        
        if not csv_files:
            raise RuntimeError(
                f"CICFlowMeter did not produce CSV. "
                f"stdout: {result.stdout}, stderr: {result.stderr}"
            )
        
        # Read the first CSV (should contain all flows)
        df = pd.read_csv(csv_files[0])
        return df
    
    def extract_first_flow(self, pcap_path: str) -> Optional[np.ndarray]:
        """
        Convert pcap and return the feature vector of the first flow only.
        
        Returns:
            Feature vector as numpy array, or None if no flows.
        """
        df = self.convert(pcap_path)
        
        if len(df) == 0:
            return None
        
        # Drop non-feature columns
        drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        first_flow = df.iloc[0][feature_cols].values.astype(float)
        return first_flow
    
    def is_available(self) -> bool:
        """Check if CICFlowMeter is installed and accessible."""
        return self.jar_path.exists()
