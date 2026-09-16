"""Host power state: whether the laptop is on mains power, and keeping it awake.

On 2026-09-14 the open-ended main stage ran on battery for 44 minutes. The GPU's
enforced limit fell from about 98 W to about 51 W, and Windows suspended the
machine twice inside running windows (protocol section 13.7). Energy measured in
that state is not comparable with the rest of the grid, so the labelling harness
waits for mains power and asks Windows not to sleep while a stage runs.

Both helpers are Windows-only and degrade to "unknown" and "no-op" elsewhere.
Keeping the system awake does not survive closing the lid, and does not stop
Windows hibernating on a critically low battery; the mains-power check covers that.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from types import TracebackType

LOGGER = logging.getLogger("arbiter.power")
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
AC_OFFLINE, AC_ONLINE = 0, 1


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def ac_online() -> bool | None:
    """Return True on mains power, False on battery, None when it cannot be read."""
    if sys.platform != "win32":
        return None
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    if status.ACLineStatus == AC_ONLINE:
        return True
    if status.ACLineStatus == AC_OFFLINE:
        return False
    return None


class KeepAwake:
    """Context manager that stops Windows from sleeping on idle while it is held.

    The execution state belongs to the calling thread, so enter and exit it on the
    thread that runs the stage.
    """

    def __enter__(self) -> KeepAwake:
        """Request that the system stays awake."""
        if sys.platform == "win32":
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            if not ctypes.windll.kernel32.SetThreadExecutionState(flags):
                LOGGER.warning("could not ask Windows to stay awake; idle sleep remains possible")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release the request."""
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
