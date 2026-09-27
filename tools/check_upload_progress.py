#!/usr/bin/env python3
"""Check exact Git LFS upload progress."""

import subprocess
import time

def main():
    try:
        cmd = ['powershell', '-Command', '(Get-CimInstance Win32_Process -Filter "ProcessId = 7436").ReadTransferCount']
        out1 = subprocess.check_output(cmd, text=True).strip()
        b1 = int(out1)
        time.sleep(2.0)
        out2 = subprocess.check_output(cmd, text=True).strip()
        b2 = int(out2)
        total = 7920627181
        pct = min(100.0, (b2 / total) * 100)
        speed = (b2 - b1) / (2.0 * 1024 * 1024)
        print(f"Uploaded: {b2:,} / {total:,} bytes ({b2/(1024**3):.2f} GB / {total/(1024**3):.2f} GB) -> {pct:.2f}% | Speed: {speed:.2f} MB/s")
    except Exception as e:
        print(f"Error checking progress (process may have completed): {e}")

if __name__ == "__main__":
    main()
