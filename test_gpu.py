import time
import os
import argparse
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetCount,
    nvmlShutdown,
)

def clear_terminal_lines(n: int):
    for _ in range(n):
        print("\033[F\033[K", end="")

def bytes_to_mb(x):
    return x / 1024 / 1024

def format_bar(used_mb, total_mb, width=25):
    ratio = min(used_mb / total_mb, 1.0)
    filled = int(ratio * width)
    bar = '█' * filled + '-' * (width - filled)
    percent = ratio * 100
    return f"[{bar}] {percent:5.1f}%"

def get_gpu_stats():
    stats = []
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        total = bytes_to_mb(mem_info.total)
        used = bytes_to_mb(mem_info.used)
        free = bytes_to_mb(mem_info.free)
        stats.append({
            "id": i,
            "name": name,
            "total": total,
            "used": used,
            "free": free,
            "bar": format_bar(used, total)
        })
    return stats

def print_gpu_stats(stats):
    print(f"{'GPU':<5}{'Name':<30}{'Total(MB)':>10}{'Used(MB)':>10}{'Free(MB)':>10}   {'Usage':<30}")
    for s in stats:
        print(f"{s['id']:<5}{s['name']:<30}{s['total']:>10.0f}{s['used']:>10.1f}{s['free']:>10.1f}   {s['bar']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.1, help="Интервал обновления (в секундах)")
    args = parser.parse_args()

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        rows_needed = device_count + 2

        print("\033c", end="")
        while True:
            stats = get_gpu_stats()
            print_gpu_stats(stats)
            time.sleep(args.interval)
            clear_terminal_lines(rows_needed)

    except KeyboardInterrupt:
        print("\nЗавершено пользователем.")
    except Exception as e:
        print(f"\nОшибка: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass

if __name__ == "__main__":
    main()
