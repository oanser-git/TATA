# QEMU/KVM Testbed Setup

This directory contains scripts to set up a two-VM QEMU/KVM testbed for TATA's RL-based traffic generation.

## Architecture

```
Host Machine (Linux with KVM)
  └── br-tata (192.168.100.1/24)  [Linux bridge]
      ├── tap-tata-client  ────>  Client VM (192.168.100.10)
      │                             ├── tc qdisc netem (traffic shaping)
      │                             └── SSH/HTTP traffic generators
      └── tap-tata-server  ────>  Server VM (192.168.100.20)
                                    ├── SSH server
                                    ├── HTTP server (nginx)
                                    └── iperf3 server
```

## Prerequisites

- Linux host with **KVM enabled** (`/dev/kvm` exists)
- `qemu-system-x86_64` installed
- `bridge-utils` (for `brctl` / `ip` bridge commands)
- `cloud-image-utils` (for `cloud-localds`)
- Root/sudo access (for bridge and tap creation)
- CICFlowMeter Java jar (for flow extraction)

## Quick Start

### 1. Download Ubuntu Cloud Images

```bash
mkdir -p /var/lib/tata
wget https://cloud-images.ubuntu.com/minimal/releases/jammy/release/ubuntu-22.04-minimal-cloudimg-amd64.img \
  -O /var/lib/tata/base.img
```

### 2. Create VM Disk Images

```bash
qemu-img create -f qcow2 -F qcow2 -b /var/lib/tata/base.img /var/lib/tata/client.qcow2 10G
qemu-img create -f qcow2 -F qcow2 -b /var/lib/tata/base.img /var/lib/tata/server.qcow2 10G
```

### 3. Build Cloud-Init ISOs

```bash
sudo apt install cloud-image-utils
./build_cloud_init.sh
# Outputs:
#   cloud-init/client.iso
#   cloud-init/server.iso
#   cloud-init/tata_vm_key     (SSH private key)
#   cloud-init/tata_vm_key.pub (SSH public key)
```

### 4. Start the Testbed

```bash
sudo ./testbed.sh start
```

This creates:
- `br-tata` bridge with IP `192.168.100.1/24`
- Two TAP interfaces attached to the bridge
- Client VM at `192.168.100.10`
- Server VM at `192.168.100.20`

Wait ~30-60 seconds for VMs to boot and configure via cloud-init.

### 5. Verify Connectivity

```bash
ssh -i cloud-init/tata_vm_key tata@192.168.100.10  # Client
ssh -i cloud-init/tata_vm_key tata@192.168.100.20  # Server
```

### 6. Use from Python

```python
from src.testbed.qemu_testbed import QemuTestbed

testbed = QemuTestbed(
    client_ip="192.168.100.10",
    server_ip="192.168.100.20",
    ssh_key_path="scripts/qemu/cloud-init/tata_vm_key",
)

# Apply traffic shaping (maps action [0,1] to tc parameters)
testbed.apply_config(action)

# Generate SSH traffic and capture pcap
pcap_path = testbed.generate_traffic(scenario="ssh")

# Convert to flow features via CICFlowMeter
flows = testbed.collect_flows(pcap_path)
```

### 7. Stop the Testbed

```bash
sudo ./testbed.sh stop
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TATA_CLIENT_IMAGE` | `/var/lib/tata/client.qcow2` | Client VM disk image |
| `TATA_SERVER_IMAGE` | `/var/lib/tata/server.qcow2` | Server VM disk image |
| `TATA_CLIENT_CLOUD_INIT` | `cloud-init/client.iso` | Client cloud-init ISO |
| `TATA_SERVER_CLOUD_INIT` | `cloud-init/server.iso` | Server cloud-init ISO |
| `TATA_VM_MEM` | `1024` | VM memory in MB |
| `TATA_VM_CPUS` | `1` | VM CPU cores |

## Troubleshooting

**VMs not reachable:**
- Check `sudo ./testbed.sh status`
- Verify bridge is up: `ip addr show br-tata`
- Check VM serial logs: `tail /var/log/tata-client.log`

**KVM not available:**
- Check BIOS virtualization settings (Intel VT-x / AMD-V)
- Load module: `sudo modprobe kvm_intel` or `sudo modprobe kvm_amd`

**tc qdisc fails:**
- Ensure client VM has `iproute2` installed (via cloud-init packages)
- Run inside VM: `sudo tc qdisc show dev eth0`

**CICFlowMeter not found:**
- Download from https://github.com/GintsEngelen/CICFlowMeter
- Build with `mvn package`
- Set path in `QemuTestbed(cicflowmeter_jar=...)`
