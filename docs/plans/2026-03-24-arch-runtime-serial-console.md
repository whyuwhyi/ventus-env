# Arch Runtime Serial Console Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Arch-based `ventus-runtime.qcow2` boot visibly and log in over QEMU serial so `make qemu` works with `-nographic`.

**Architecture:** Keep the existing qcow2 disk boot flow, but configure the guest as a serial-first appliance. Update GRUB to emit to serial and pass `console=ttyS0` to the kernel, then enable `serial-getty@ttyS0.service` so login stays on the same device.

**Tech Stack:** QEMU, qcow2, Arch Linux guest, GRUB, systemd, guestfish

---

### Task 1: Make boot output reach the serial console

**Files:**
- Modify in guest image: `/etc/default/grub`
- Modify in guest image: `/boot/grub/grub.cfg`

**Step 1: Confirm the current failure mode**

Run: `make qemu`
Expected: `SeaBIOS` and `GRUB` text appear on the terminal, then progress becomes invisible or appears stuck.

**Step 2: Write the minimal config**

Set GRUB to:
- `GRUB_TERMINAL_INPUT=serial console`
- `GRUB_TERMINAL_OUTPUT=serial console`
- `GRUB_SERIAL_COMMAND="serial --speed=115200 --unit=0 --word=8 --parity=no --stop=1"`
- append `console=ttyS0,115200 console=tty0` to `GRUB_CMDLINE_LINUX`

**Step 3: Regenerate or patch active GRUB config**

Ensure the guest bootloader actually uses those settings by regenerating `grub.cfg` or editing the generated file as a fallback.

### Task 2: Make login visible on the serial console

**Files:**
- Enable in guest image: `serial-getty@ttyS0.service`

**Step 1: Enable serial getty**

Create the systemd enablement symlink for `serial-getty@ttyS0.service`.

**Step 2: Keep existing console behavior**

Do not remove `tty1` login or graphical defaults. Add serial access only.

### Task 3: Verify the runtime entrypoint

**Files:**
- Verify host file: `ventus-pytorch/ventus-runtime/Makefile`

**Step 1: Boot the updated image**

Run: `make qemu`
Expected: kernel boot messages and a login prompt appear directly in the terminal.

**Step 2: Preserve the final deliverable shape**

Keep:
- `ventus-pytorch/ventus-runtime/ventus-runtime.qcow2`
- `ventus-pytorch/ventus-runtime/Makefile`

Do not introduce extra runtime entrypoints.
