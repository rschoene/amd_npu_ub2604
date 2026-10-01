#!/usr/bin/env bash
#
# setup_npu.sh — AMD NPU (XDNA) software stack installer
#
# Usage:
#   ./setup_npu.sh install    Install the full NPU software stack
#   ./setup_npu.sh uninstall  Remove everything this script installed
#   ./setup_npu.sh status     Show current NPU stack status
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MEMLOCK_CONF="/etc/security/limits.d/99-npu-memlock.conf"
SYSTEMD_DROPIN_DIR="/etc/systemd/system/user@.service.d"
SYSTEMD_DROPIN="${SYSTEMD_DROPIN_DIR}/99-npu-memlock.conf"
MEMLOCK_KB=1048576  # 1 GB

XRT_PACKAGES=(
    libxrt2
    libxrt-npu2
    libxrt-utils-npu
    python3-xrt
    libxrt-dev
)

# ─── Helpers ───────────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[FAIL]\033[0m  %s\n' "$*" >&2; }

need_sudo() {
    if [[ $EUID -ne 0 ]]; then
        SUDO="sudo"
    else
        SUDO=""
    fi
}

# Discover the accel device path from sysfs (e.g. /dev/accel/accel0)
get_accel_dev_path() {
    local uevent
    for dev in /sys/class/accel/accel*; do
        [[ -e "$dev/uevent" ]] || continue
        uevent=$(grep "^DEVNAME=" "$dev/uevent" 2>/dev/null | cut -d= -f2)
        if [[ -n "$uevent" ]]; then
            echo "/dev${uevent}"
            return 0
        fi
    done
    return 1
}

# ─── Install ───────────────────────────────────────────────────────────────────

do_install() {
    need_sudo
    local changed=0

    # Authenticate sudo once upfront
    if [[ $EUID -ne 0 ]]; then
        info "This will require sudo. Enter password if prompted."
        $SUDO -v
    fi

    # 1. Kernel driver check
    info "Checking kernel driver (amdxdna)..."
    if lsmod | grep -q amdxdna; then
        ok "amdxdna driver is loaded"
    else
        warn "amdxdna driver not loaded — trying to load it"
        # Load dependencies first
        $SUDO modprobe gpu_sched 2>/dev/null || true
        $SUDO modprobe amd_pmf 2>/dev/null || true
        $SUDO modprobe amdxdna
        sleep 2
        if lsmod | grep -q amdxdna; then
            ok "amdxdna driver loaded"
        elif [[ -n "$(get_accel_dev_path)" ]]; then
            ok "amdxdna driver active (device present)"
        else
            err "Could not load amdxdna driver"
            err "Last kernel messages:"
            $SUDO dmesg | grep -i "xdna\|npu\|accel" | tail -5
            err "Is the NPU present?"
            lspci | grep -i "neural\|npu" || true
            return 1
        fi
    fi

    # 2. Firmware check
    info "Checking NPU firmware..."
    if [[ -d /lib/firmware/amdnpu ]]; then
        ok "NPU firmware present: $(ls /lib/firmware/amdnpu/ | tr '\n' ' ')"
    else
        warn "NPU firmware not found — installing linux-firmware-amd-misc"
        $SUDO apt-get install -y linux-firmware-amd-misc
        changed=1
    fi

    # 3. XRT packages
    info "Checking XRT packages..."
    local missing=()
    for pkg in "${XRT_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        info "Installing: ${missing[*]}"
        $SUDO apt-get install -y "${missing[@]}"
        changed=1
    else
        ok "All XRT packages already installed"
    fi

    # 4. Memlock limit
    # On modern Ubuntu, systemd's DefaultLimitMEMLOCK overrides PAM limits,
    # so we need to set it in BOTH places.
    info "Setting memlock limit to ${MEMLOCK_KB} KB (1 GB)..."

    # 4a. PAM limits (for login sessions)
    if [[ -f "$MEMLOCK_CONF" ]] && grep -q "memlock" "$MEMLOCK_CONF"; then
        ok "PAM memlock limit already configured"
    else
        echo "$USER memlock ${MEMLOCK_KB}" | $SUDO tee "$MEMLOCK_CONF" > /dev/null
        ok "PAM memlock limit configured"
        changed=1
    fi

    # 4b. systemd drop-in for user@.service (takes precedence on systemd logins)
    if [[ -f "$SYSTEMD_DROPIN" ]] && grep -q "LimitMEMLOCK" "$SYSTEMD_DROPIN"; then
        ok "systemd memlock drop-in already configured"
    else
        $SUDO mkdir -p "$SYSTEMD_DROPIN_DIR"
        {
            echo "[Service]"
            echo "LimitMEMLOCK=${MEMLOCK_KB}K"
        } | $SUDO tee "$SYSTEMD_DROPIN" > /dev/null
        $SUDO systemctl daemon-reload
        ok "systemd memlock drop-in configured"
        changed=1
    fi

    # 5. Python venv + ONNX Runtime
    info "Setting up Python virtual environment..."
    if ! dpkg -s python3-venv &>/dev/null; then
        info "Installing python3-venv..."
        $SUDO apt-get install -y python3-venv
        changed=1
    fi
    # Remove broken venv (no python or no pip)
    if [[ -d "$VENV_DIR" && (! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/pip") ]]; then
        warn "Removing broken venv (missing python or pip)..."
        rm -rf "$VENV_DIR"
    fi
    # Recreate if it lacks system-site-packages (needed to see pyxrt)
    if [[ -d "$VENV_DIR" ]] && ! grep -q "system-site-packages" "$VENV_DIR/pyvenv.cfg" 2>/dev/null; then
        warn "Recreating venv with --system-site-packages (for pyxrt access)..."
        rm -rf "$VENV_DIR"
    fi
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv --system-site-packages "$VENV_DIR"
        # If pip still missing, bootstrap it
        if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
            "$VENV_DIR/bin/python" -m ensurepip --upgrade 2>/dev/null || {
                "$VENV_DIR/bin/python" -m pip --version 2>/dev/null || {
                    warn "pip not available in venv, trying get-pip.py..."
                    curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python"
                }
            }
        fi
        ok "Created venv at $VENV_DIR"
    else
        ok "Venv already exists at $VENV_DIR"
    fi

    if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
        err "pip is not available in the venv. Cannot continue."
        return 1
    fi

    info "Installing ONNX Runtime into venv..."
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet onnxruntime
    ok "ONNX Runtime installed: $("${VENV_DIR}/bin/python" -c 'import onnxruntime; print(onnxruntime.__version__)')"

    # 6. Summary
    echo
    if [[ $changed -eq 1 ]]; then
        warn "System packages were modified. You may need to log out and back in"
        warn "for the memlock limit to take effect."
    fi
    ok "NPU software stack installation complete."
    echo
    echo "  To verify:  ./setup_npu.sh status"
    echo "  To use:     source .venv/bin/activate"
    echo "  To test:    .venv/bin/python scripts/test_npu.py"
}

# ─── Uninstall ─────────────────────────────────────────────────────────────────

do_uninstall() {
    need_sudo
    local removed=0

    # 1. Remove memlock configs
    if [[ -f "$MEMLOCK_CONF" ]]; then
        info "Removing PAM memlock config: $MEMLOCK_CONF"
        $SUDO rm -f "$MEMLOCK_CONF"
        ok "Removed"
        removed=1
    else
        info "PAM memlock config not found (already clean)"
    fi

    if [[ -f "$SYSTEMD_DROPIN" ]]; then
        info "Removing systemd memlock drop-in: $SYSTEMD_DROPIN"
        $SUDO rm -f "$SYSTEMD_DROPIN"
        # Remove the drop-in dir if empty
        $SUDO rmdir "$SYSTEMD_DROPIN_DIR" 2>/dev/null || true
        $SUDO systemctl daemon-reload
        ok "Removed"
        removed=1
    else
        info "systemd memlock drop-in not found (already clean)"
    fi

    # 2. Remove venv
    if [[ -d "$VENV_DIR" ]]; then
        info "Removing virtual environment: $VENV_DIR"
        rm -rf "$VENV_DIR"
        ok "Removed"
        removed=1
    else
        info "Venv not found (already clean)"
    fi

    # 3. Remove XRT packages (optional — ask)
    echo
    read -rp "Remove XRT apt packages as well? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        info "Removing XRT packages: ${XRT_PACKAGES[*]}"
        $SUDO apt-get remove -y "${XRT_PACKAGES[@]}"
        $SUDO apt-get autoremove -y
        ok "XRT packages removed"
        removed=1
    else
        info "Keeping XRT packages"
    fi

    # 4. Summary
    echo
    if [[ $removed -eq 1 ]]; then
        ok "Uninstall complete."
        warn "Log out and back in for memlock changes to take effect."
    else
        ok "Nothing to remove."
    fi
}

# ─── Status ────────────────────────────────────────────────────────────────────

do_status() {
    echo "═══════════════════════════════════════════════"
    echo "  AMD NPU Stack Status"
    echo "═══════════════════════════════════════════════"
    echo

    # Hardware
    echo "  Hardware:"
    local npu_pci
    npu_pci=$(lspci 2>/dev/null | grep -i "neural processing" || echo "  NOT FOUND")
    echo "    $npu_pci"

    # Driver
    echo
    echo "  Kernel driver:"
    if lsmod | grep -q amdxdna; then
        local fw
        fw=$(cat /sys/class/accel/accel0/device/fw_version 2>/dev/null || echo "?")
        local pwr
        pwr=$(cat /sys/class/accel/accel0/device/power/runtime_status 2>/dev/null || echo "?")
        ok "amdxdna loaded (fw: $fw, power: $pwr)"
    else
        err "amdxdna NOT loaded"
    fi

    # Device node (discovered from sysfs)
    echo
    echo "  Device node:"
    local dev_path
    dev_path=$(get_accel_dev_path)
    if [[ -n "$dev_path" && -e "$dev_path" ]]; then
        ok "$dev_path exists ($(stat -c '%A %U:%G' "$dev_path"))"
    else
        err "No accel device node found in /sys/class/accel/"
    fi

    # Firmware
    echo
    echo "  Firmware:"
    if [[ -d /lib/firmware/amdnpu ]]; then
        ok "amdnpu firmware: $(ls /lib/firmware/amdnpu/ | tr '\n' ' ')"
    else
        err "NPU firmware NOT installed"
    fi

    # XRT
    echo
    echo "  XRT (userspace):"
    local xrt_ok=1
    for pkg in "${XRT_PACKAGES[@]}"; do
        if dpkg -s "$pkg" &>/dev/null 2>&1; then
            local ver
            ver=$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null)
            echo "    ✓ $pkg ($ver)"
        else
            echo "    ✗ $pkg (missing)"
            xrt_ok=0
        fi
    done
    [[ $xrt_ok -eq 1 ]] && ok "All XRT packages present" || warn "Some XRT packages missing"

    # Memlock
    echo
    echo "  Memlock limit:"
    local current
    current=$(ulimit -l)
    if [[ -f "$MEMLOCK_CONF" ]]; then
        echo "    PAM config: $(cat "$MEMLOCK_CONF")"
    else
        echo "    PAM config: NOT SET"
    fi
    if [[ -f "$SYSTEMD_DROPIN" ]]; then
        echo "    systemd: $(grep LimitMEMLOCK "$SYSTEMD_DROPIN")"
    else
        echo "    systemd: NOT SET"
    fi
    echo "    Current session: ${current} KB"
    if [[ "$current" -ge 65536 ]]; then
        ok "Sufficient for NPU (needs ≥ 64 MB)"
    else
        warn "Too low! NPU needs ≥ 64 MB."
        warn "Run install, then reboot (or: sudo systemctl daemon-reload && re-login)"
    fi

    # Venv
    echo
    echo "  Python environment:"
    if [[ -d "$VENV_DIR" ]]; then
        local ort_ver
        ort_ver=$("$VENV_DIR/bin/python" -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo "not installed")
        echo "    Venv: $VENV_DIR"
        echo "    ONNX Runtime: $ort_ver"
    else
        echo "    Venv: NOT CREATED"
    fi

    echo
    echo "═══════════════════════════════════════════════"
}

# ─── Main ──────────────────────────────────────────────────────────────────────

case "${1:-}" in
    install)   do_install ;;
    uninstall) do_uninstall ;;
    status)    do_status ;;
    *)
        echo "Usage: $0 {install|uninstall|status}"
        exit 1
        ;;
esac
