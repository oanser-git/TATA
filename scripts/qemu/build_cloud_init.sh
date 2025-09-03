#!/bin/bash
# Build cloud-init ISO images for VM provisioning.
# Requires cloud-localds (from cloud-utils package).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/cloud-init"
mkdir -p "${OUTPUT_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Generate SSH key pair for VM access
SSH_KEY="${OUTPUT_DIR}/tata_vm_key"
if [[ ! -f "${SSH_KEY}" ]]; then
    log "Generating SSH key pair for VM access..."
    ssh-keygen -t ed25519 -f "${SSH_KEY}" -N "" -C "tata-testbed"
fi

PUB_KEY="$(cat ${SSH_KEY}.pub)"

# --- Client cloud-init ---
log "Creating client cloud-init..."
cat > "${OUTPUT_DIR}/client-user-data.yaml" <<EOF
#cloud-config
hostname: tata-client
users:
  - name: tata
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ${PUB_KEY}
    shell: /bin/bash
packages:
  - iperf3
  - ssh
  - curl
  - iproute2
  - tcpdump
  - net-tools
write_files:
  - path: /etc/netplan/01-netcfg.yaml
    content: |
      network:
        version: 2
        ethernets:
          eth0:
            dhcp4: no
            addresses:
              - 192.168.100.10/24
            gateway4: 192.168.100.1
            nameservers:
              addresses: [8.8.8.8]
    permissions: '0644'
runcmd:
  - netplan apply
  - systemctl enable ssh
  - systemctl start ssh
  - echo "Client ready" > /var/run/tata-ready
EOF

cat > "${OUTPUT_DIR}/client-meta-data.yaml" <<EOF
instance-id: tata-client
local-hostname: tata-client
EOF

cloud-localds "${OUTPUT_DIR}/client.iso" \
    "${OUTPUT_DIR}/client-user-data.yaml" \
    "${OUTPUT_DIR}/client-meta-data.yaml" || {
    log "ERROR: cloud-localds failed. Install cloud-utils (e.g., apt install cloud-image-utils)."
    exit 1
}

# --- Server cloud-init ---
log "Creating server cloud-init..."
cat > "${OUTPUT_DIR}/server-user-data.yaml" <<EOF
#cloud-config
hostname: tata-server
users:
  - name: tata
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - ${PUB_KEY}
    shell: /bin/bash
packages:
  - openssh-server
  - nginx
  - dnsmasq
  - iperf3
  - net-tools
write_files:
  - path: /etc/netplan/01-netcfg.yaml
    content: |
      network:
        version: 2
        ethernets:
          eth0:
            dhcp4: no
            addresses:
              - 192.168.100.20/24
            gateway4: 192.168.100.1
            nameservers:
              addresses: [8.8.8.8]
    permissions: '0644'
runcmd:
  - netplan apply
  - systemctl enable ssh
  - systemctl start ssh
  - systemctl enable nginx
  - systemctl start nginx
  - echo "Server ready" > /var/run/tata-ready
EOF

cat > "${OUTPUT_DIR}/server-meta-data.yaml" <<EOF
instance-id: tata-server
local-hostname: tata-server
EOF

cloud-localds "${OUTPUT_DIR}/server.iso" \
    "${OUTPUT_DIR}/server-user-data.yaml" \
    "${OUTPUT_DIR}/server-meta-data.yaml"

log "Cloud-init ISOs built:"
log "  ${OUTPUT_DIR}/client.iso"
log "  ${OUTPUT_DIR}/server.iso"
log "SSH key: ${SSH_KEY}"
