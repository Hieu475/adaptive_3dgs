"""Low-overhead optional NVML sampler for GPU timing diagnostics.

The sampler records timestamped device state outside the measured CUDA event
interval. Unsupported NVML fields are retained as ``None`` instead of guessed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import statistics
import threading
import time


class GPUTelemetrySampler:
    def __init__(self, device_index: int = 0, interval_ms: float = 50.0):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        library = ctypes.util.find_library("nvidia-ml")
        if not library:
            raise RuntimeError("NVML library libnvidia-ml was not found")
        self._lib = ctypes.CDLL(library)
        self._handle = ctypes.c_void_p()
        self.interval_s = interval_ms / 1000.0
        self._samples: list[dict] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._enabled.set()
        self._sampling = False
        self._thread: threading.Thread | None = None
        self._bind_and_initialize(device_index)

    def _function(self, name, restype, argtypes):
        fn = getattr(self._lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def _bind_and_initialize(self, device_index: int):
        rc = ctypes.c_int
        u = ctypes.c_uint
        h = ctypes.c_void_p
        p_u = ctypes.POINTER(u)
        self._init = self._function("nvmlInit_v2", rc, [])
        self._shutdown = self._function("nvmlShutdown", rc, [])
        self._get_handle = self._function("nvmlDeviceGetHandleByIndex_v2", rc,
                                          [u, ctypes.POINTER(h)])
        self._init()
        self._get_handle(device_index, ctypes.byref(self._handle))
        self._get_util = self._function("nvmlDeviceGetUtilizationRates", rc,
                                        [h, ctypes.c_void_p])
        self._get_memory = self._function("nvmlDeviceGetMemoryInfo", rc,
                                          [h, ctypes.c_void_p])
        self._get_clock = self._function("nvmlDeviceGetClockInfo", rc, [h, u, p_u])
        self._get_temp = self._function("nvmlDeviceGetTemperature", rc, [h, u, p_u])
        self._get_pstate = self._function("nvmlDeviceGetPerformanceState", rc, [h, p_u])
        self._get_power = self._function("nvmlDeviceGetPowerUsage", rc, [h, p_u])

    @staticmethod
    def _read_u(call, *args) -> int | None:
        value = ctypes.c_uint()
        try:
            rc = call(*args, ctypes.byref(value))
            return int(value.value) if rc == 0 else None
        except Exception:
            return None

    def _sample(self) -> dict:
        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]

        util = Utilization()
        memory = Memory()
        try:
            util_rc = self._get_util(self._handle, ctypes.byref(util))
            util_gpu = int(util.gpu) if util_rc == 0 else None
            util_memory = int(util.memory) if util_rc == 0 else None
        except Exception:
            util_gpu = util_memory = None
        try:
            memory_rc = self._get_memory(self._handle, ctypes.byref(memory))
            memory_used = int(memory.used) if memory_rc == 0 else None
        except Exception:
            memory_used = None
        return {
            "timestamp_unix_ns": time.time_ns(),
            "gpu_utilization_percent": util_gpu,
            "memory_utilization_percent": util_memory,
            "memory_used_bytes": memory_used,
            "graphics_clock_mhz": self._read_u(self._get_clock, self._handle, 0),
            "memory_clock_mhz": self._read_u(self._get_clock, self._handle, 2),
            "temperature_c": self._read_u(self._get_temp, self._handle, 0),
            "performance_state": self._read_u(self._get_pstate, self._handle),
            "power_mw": self._read_u(self._get_power, self._handle),
        }

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            if not self._enabled.wait(timeout=0.1):
                continue
            with self._condition:
                if not self._enabled.is_set():
                    continue
                self._sampling = True
            try:
                sample = self._sample()
            except Exception as exc:
                sample = {"timestamp_unix_ns": time.time_ns(), "sample_error": repr(exc)}
            finally:
                with self._condition:
                    self._sampling = False
                    self._condition.notify_all()
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def pause(self):
        self._enabled.clear()
        with self._condition:
            self._condition.wait_for(lambda: not self._sampling, timeout=2.0)

    def resume(self):
        self._enabled.set()

    def window(self, start_unix_ns: int | None, end_unix_ns: int | None) -> dict | None:
        if start_unix_ns is None or end_unix_ns is None:
            return None
        with self._lock:
            samples = list(self._samples)
        before = [s for s in samples if s["timestamp_unix_ns"] < start_unix_ns]
        inside = [s for s in samples if start_unix_ns <= s["timestamp_unix_ns"] <= end_unix_ns]
        after = [s for s in samples if s["timestamp_unix_ns"] > end_unix_ns]
        selected = []
        if before:
            selected.append(before[-1])
        selected.extend(inside)
        if after:
            selected.append(after[0])
        fields = ("gpu_utilization_percent", "memory_utilization_percent",
                  "memory_used_bytes", "graphics_clock_mhz", "memory_clock_mhz",
                  "temperature_c", "performance_state", "power_mw")
        stats = {}
        for field in fields:
            values = [s[field] for s in inside if s[field] is not None]
            if values and all(isinstance(x, (int, float)) for x in values):
                stats[field] = {"mean": statistics.mean(values),
                                "min": min(values), "max": max(values)}
            elif values:
                stats[field] = {"values": sorted(set(values))}
            else:
                stats[field] = None
        return {"sample_interval_ms": self.interval_s * 1000,
                "samples_inside_stage": len(inside),
                "sample_count_in_window": len(selected),
                "window_stats": stats,
                "samples": selected}

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._enabled.set()
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
            self._thread = None
        try:
            self._shutdown()
        except Exception:
            pass

