#!/usr/bin/env python3
"""
test_npu.py — AMD NPU (XDNA) diagnostic and test script

Checks:
  1. Memlock limit (needs >= 64 MB)
  2. Device node access
  3. XRT device enumeration
  4. NPU device open
  5. Basic device info

Usage:
  .venv/bin/python scripts/test_npu.py
"""

import os
import sys
import resource


def check_memlock():
    """Check if memlock limit is sufficient for NPU."""
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    soft_mb = soft / (1024 * 1024)
    hard_mb = hard / (1024 * 1024)
    print(f"  Memlock limit: soft={soft_mb:.1f} MB, hard={hard_mb:.1f} MB")

    if soft < 64 * 1024 * 1024:  # 64 MB
        print("  [FAIL] Memlock too low! NPU needs >= 64 MB.")
        print("         Log out and back in after running setup_npu.sh install")
        return False
    print("  [ OK ] Memlock limit sufficient")
    return True


def find_accel_device():
    """Discover the accel device path from sysfs."""
    accel_dir = "/sys/class/accel"
    if not os.path.isdir(accel_dir):
        return None
    for entry in sorted(os.listdir(accel_dir)):
        uevent_path = os.path.join(accel_dir, entry, "uevent")
        if not os.path.isfile(uevent_path):
            continue
        with open(uevent_path) as f:
            for line in f:
                if line.startswith("DEVNAME="):
                    devname = line.strip().split("=", 1)[1]
                    return f"/dev/{devname}"
    return None


def check_device_node():
    """Check if the NPU device node exists and is accessible."""
    dev_path = find_accel_device()
    if dev_path is None:
        print("  [FAIL] No accel device found in /sys/class/accel/")
        print("         Is the amdxdna driver loaded? (lsmod | grep amdxdna)")
        return False

    print(f"  Device node: {dev_path}")

    if not os.path.exists(dev_path):
        print("  [FAIL] Device node not found!")
        return False

    if not os.access(dev_path, os.R_OK | os.W_OK):
        print("  [FAIL] No read/write access to device node")
        print(f"         Permissions: {oct(os.stat(dev_path).st_mode)}")
        return False

    print("  [ OK ] Device node exists and is accessible")
    return True


def check_xrt():
    """Check XRT device enumeration and NPU access."""
    try:
        import pyxrt
    except ImportError:
        print("  [FAIL] pyxrt not available")
        return False

    print(f"  pyxrt version: {getattr(pyxrt, '__version__', 'unknown')}")

    # Enumerate devices
    try:
        count = pyxrt.enumerate_devices()
        print(f"  XRT devices found: {count}")
        if count == 0:
            print("  [FAIL] No XRT devices found")
            return False
        print("  [ OK ] XRT device enumeration successful")
    except Exception as e:
        print(f"  [FAIL] Device enumeration error: {e}")
        return False

    # Try to open the NPU device
    try:
        dev = pyxrt.device(0)
        print(f"  [ OK ] NPU device opened: {dev}")

        # Try to get device info
        try:
            info = dev.get_info()
            print(f"  Device info: {info}")
        except Exception:
            pass

        # List available methods
        methods = [m for m in dir(dev) if not m.startswith('_') and callable(getattr(dev, m))]
        print(f"  Available methods: {methods}")

        return True
    except Exception as e:
        print(f"  [FAIL] Could not open NPU device: {e}")
        if "mmap" in str(e) and "Resource temporarily unavailable" in str(e):
            print("         This is likely a memlock limit issue.")
            print("         Log out and back in to apply the new limit.")
        return False


def main():
    print("=" * 50)
    print("  AMD NPU (XDNA) Diagnostic Test")
    print("=" * 50)

    all_ok = True

    print("\n[1] Memlock limit:")
    all_ok &= check_memlock()

    print("\n[2] Device node:")
    all_ok &= check_device_node()

    print("\n[3] XRT / NPU access:")
    all_ok &= check_xrt()

    print("\n" + "=" * 50)
    if all_ok:
        print("  ALL CHECKS PASSED — NPU is ready!")
        print("=" * 50)
        return 0
    else:
        print("  SOME CHECKS FAILED — see above for details")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
