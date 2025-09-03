#!/bin/bash
# TATA QEMU/KVM Testbed Setup Script
# Creates bridge, tap interfaces, and launches two VMs.
# Must be run as root.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_NAME="br-tata"
CLIENT_TAP="tap-tata-client"
SERVER_TAP="tap-tata-server"
SUBNET="192.168.100"
CLIENT_IP="${SUBNET}.10"
SERVER_IP="${SUBNET}.20"
NETMASK="255.255.255.0"

# VM image paths (set these before running)
CLIENT_IMAGE="${TATA_CLIENT_IMAGE:-/var/lib/tata/client.qcow2}"
SERVER_IMAGE="${TATA_SERVER_IMAGE:-/var/lib/tata/server.qcow2}"
CLIENT_CLOUD_INIT="${TATA_CLIENT_CLOUD_INIT:-${SCRIPT_DIR}/cloud-init/client.iso}"
SERVER_CLOUD_INIT="${TATA_SERVER_CLOUD_INIT:-${SCRIPT_DIR}/cloud-init/server.iso}"

# VM specs
VM_MEM="${TATA_VM_MEM:-1024}"
VM_CPUS="${TATA_VM_CPUS:-1}"

# --- Helper functions ---

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "ERROR: This script must be run as root"
        exit 1
    fi
}

check_kvm() {
    if [[ ! -e /dev/kvm ]]; then
        echo "ERROR: /dev/kvm not found. Ensure KVM is enabled in BIOS and kernel module is loaded."
        exit 1
    fi
    log "KVM available: OK"
}

create_bridge() {
    log "Creating bridge ${BRIDGE_NAME}..."
    if ip link show "${BRIDGE_NAME}" &>/dev/null; then
        log "Bridge ${BRIDGE_NAME} already exists."
    else
        ip link add name "${BRIDGE_NAME}" type bridge
        ip addr add "${SUBNET}.1/24" dev "${BRIDGE_NAME}"
        ip link set "${BRIDGE_NAME}" up
        log "Bridge ${BRIDGE_NAME} created with IP ${SUBNET}.1/24"
    fi
}

create_tap() {
    local tap_name="$1"
    log "Creating tap interface ${tap_name}..."
    if ip link show "${tap_name}" &>/dev/null; then
        log "Tap ${tap_name} already exists."
    else
        ip tuntap add dev "${tap_name}" mode tap user "${SUDO_USER:-root}"
        ip link set "${tap_name}" master "${BRIDGE_NAME}"
        ip link set "${tap_name}" up
        log "Tap ${tap_name} created and attached to ${BRIDGE_NAME}"
    fi
}

launch_vm() {
    local name="$1"
    local image="$2"
    local cloud_init="$3"
    local tap="$4"
    local mac="$5"
    local pid_file="/var/run/tata-${name}.pid"
    
    log "Launching VM: ${name}..."
    
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        log "VM ${name} already running (PID $(cat ${pid_file}))."
        return
    fi
    
    if [[ ! -f "${image}" ]]; then
        echo "ERROR: VM image not found: ${image}"
        echo "Set TATA_CLIENT_IMAGE and TATA_SERVER_IMAGE environment variables."
        exit 1
    fi
    
    local cdrom_opts=""
    if [[ -f "${cloud_init}" ]]; then
        cdrom_opts="-cdrom ${cloud_init}"
    fi
    
    qemu-system-x86_64 \
        -enable-kvm \
        -cpu host \
        -smp "${VM_CPUS}" \
        -m "${VM_MEM}" \
        -drive file="${image}",format=qcow2,if=virtio \
        ${cdrom_opts} \
        -netdev tap,id=net0,ifname="${tap}",script=no,downscript=no \
        -device virtio-net-pci,netdev=net0,mac="${mac}" \
        -daemonize \
        -pidfile "${pid_file}" \
        -display none \
        -serial file:/var/log/tata-${name}.log \
        -name "tata-${name}" \
        || { echo "ERROR: Failed to launch ${name}"; exit 1; }
    
    log "VM ${name} launched (PID file: ${pid_file})"
}

wait_for_vm() {
    local ip="$1"
    local max_wait=120
    local waited=0
    log "Waiting for VM at ${ip} to respond to ping..."
    while ! ping -c 1 -W 1 "${ip}" &>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        if [[ ${waited} -ge ${max_wait} ]]; then
            echo "ERROR: VM at ${ip} did not become reachable within ${max_wait}s"
            exit 1
        fi
    done
    log "VM at ${ip} is reachable after ${waited}s"
}

# --- Main ---

case "${1:-start}" in
    start)
        check_root
        check_kvm
        create_bridge
        create_tap "${CLIENT_TAP}"
        create_tap "${SERVER_TAP}"
        
        # Launch VMs
        launch_vm "client" "${CLIENT_IMAGE}" "${CLIENT_CLOUD_INIT}" "${CLIENT_TAP}" "52:54:00:12:34:56"
        launch_vm "server" "${SERVER_IMAGE}" "${SERVER_CLOUD_INIT}" "${SERVER_TAP}" "52:54:00:12:34:57"
        
        # Wait for VMs
        wait_for_vm "${CLIENT_IP}"
        wait_for_vm "${SERVER_IP}"
        
        log "Testbed is ready!"
        log "  Client: ${CLIENT_IP}"
        log "  Server: ${SERVER_IP}"
        ;;
    
    stop)
        check_root
        log "Stopping VMs..."
        for name in client server; do
            pid_file="/var/run/tata-${name}.pid"
            if [[ -f "${pid_file}" ]]; then
                pid=$(cat "${pid_file}")
                if kill -0 "${pid}" 2>/dev/null; then
                    log "Stopping tata-${name} (PID ${pid})..."
                    kill "${pid}"
                    wait "${pid}" 2>/dev/null || true
                fi
                rm -f "${pid_file}"
            fi
        done
        
        log "Removing tap interfaces..."
        for tap in "${CLIENT_TAP}" "${SERVER_TAP}"; do
            if ip link show "${tap}" &>/dev/null; then
                ip link set "${tap}" down
                ip tuntap del dev "${tap}" mode tap
            fi
        done
        
        log "Removing bridge..."
        if ip link show "${BRIDGE_NAME}" &>/dev/null; then
            ip link set "${BRIDGE_NAME}" down
            ip link del "${BRIDGE_NAME}"
        fi
        
        log "Testbed stopped."
        ;;
    
    status)
        echo "Bridge:"
        ip link show "${BRIDGE_NAME}" 2>/dev/null || echo "  ${BRIDGE_NAME}: not found"
        echo ""
        echo "TAP interfaces:"
        ip link show "${CLIENT_TAP}" 2>/dev/null || echo "  ${CLIENT_TAP}: not found"
        ip link show "${SERVER_TAP}" 2>/dev/null || echo "  ${SERVER_TAP}: not found"
        echo ""
        echo "VMs:"
        for name in client server; do
            pid_file="/var/run/tata-${name}.pid"
            if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
                echo "  tata-${name}: running (PID $(cat ${pid_file}))"
            else
                echo "  tata-${name}: not running"
            fi
        done
        ;;
    
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
