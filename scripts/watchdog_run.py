#!/usr/bin/env python3
"""通用进程 watchdog —— 解决"跑到一半卡住几小时"的问题。

【做什么】
- 启动你指定的命令；持续监控其 stdout/stderr 是否有新输出
- 若超过 --idle-timeout 秒没有新输出，就判定为"卡死"，杀进程组并重启
- 最多重试 --max-retries 次；首次成功立即返回 0

【用法】
    # 把 watchdog 套在任何长时命令上：
    python scripts/watchdog_run.py \
        --idle-timeout 180 --max-retries 3 \
        --log logs/runs/<id>/sft.log \
        -- python scripts/sft_train.py --config config/sft_config.yaml

    # 注意 -- 后面是被监控的命令，原样转发；进程组会被一起杀
    # 退出码：被监控命令最后一次的 exit code；watchdog 自身错误为 99

【为什么用进程组】
PyTorch / DataLoader 经常会 fork 一堆 worker 子进程；只 kill 父进程时它们留下变成
僵尸或卡住 GPU。我们用 setsid 让被监控命令成为新会话首领，然后 killpg 整组。

【输出文件】
- 持续把命令 stdout/stderr 写入 --log
- 写一个 .watchdog.json 同目录小文件，记录每次重试 / kill 时间
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_event(meta_path: Path, event: dict):
    log = []
    if meta_path.exists():
        try:
            log = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            log = []
    log.append(event)
    meta_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(cmd: list[str], log_path: Path, idle_timeout: int,
             absolute_timeout: int | None) -> tuple[int, str]:
    """启动子进程并监控。返回 (exit_code, reason)。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", buffering=1, encoding="utf-8", errors="replace")

    log_f.write(f"\n===== watchdog start cmd @ {_ts()}: {' '.join(cmd)} =====\n")
    log_f.flush()

    # setsid → 自己一个会话/进程组，方便 killpg 整组
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    pgid = os.getpgid(proc.pid) if hasattr(os, "getpgid") else None

    last_output_at = time.time()
    started_at = last_output_at
    lock = threading.Lock()

    def reader():
        nonlocal last_output_at
        try:
            for line in iter(proc.stdout.readline, ""):
                if line == "" and proc.poll() is not None:
                    break
                log_f.write(line)
                log_f.flush()
                # 同时在 watchdog 自己的 stdout 输出（让上层 tee/CI 能看到进度）
                sys.stdout.write(line)
                sys.stdout.flush()
                with lock:
                    last_output_at = time.time()
        except Exception as e:  # noqa: BLE001
            log_f.write(f"\n[watchdog reader error] {e}\n")

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    reason = "ok"
    while True:
        time.sleep(5)
        rc = proc.poll()
        if rc is not None:
            t.join(timeout=2)
            log_f.write(f"\n===== watchdog: child exited rc={rc} @ {_ts()} =====\n")
            return rc, "exited"

        with lock:
            idle = time.time() - last_output_at
        elapsed = time.time() - started_at

        if idle > idle_timeout:
            reason = f"idle {idle:.0f}s > {idle_timeout}s"
            break
        if absolute_timeout and elapsed > absolute_timeout:
            reason = f"absolute timeout {elapsed:.0f}s > {absolute_timeout}s"
            break

    # 卡死 / 超时：杀整组
    log_f.write(f"\n===== watchdog: KILL ({reason}) @ {_ts()} =====\n")
    log_f.flush()
    try:
        if pgid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            if pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
    except ProcessLookupError:
        pass
    t.join(timeout=2)
    rc = proc.poll() if proc.poll() is not None else -1
    return rc, reason


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--idle-timeout", type=int, default=180,
                    help="无新输出多少秒判定卡死并 kill（默认 180s = 3 分钟）")
    ap.add_argument("--absolute-timeout", type=int, default=0,
                    help="单次运行最长允许秒数；0 表示无限")
    ap.add_argument("--max-retries", type=int, default=3, help="kill 后最多重试几次")
    ap.add_argument("--retry-sleep", type=int, default=10)
    ap.add_argument("--log", required=True, help="命令的输出会追加到此 log；watchdog 元数据写到 <log>.watchdog.json")
    ap.add_argument("--tag", default=None, help="可选标签写到元数据，便于事后定位")
    # 把 -- 之后的全部当成被监控命令
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- 之后的命令")
    args = ap.parse_args()

    if not args.rest:
        print("❌ 请在 -- 之后给出要监控的命令", file=sys.stderr)
        sys.exit(99)
    if args.rest[0] == "--":
        cmd = args.rest[1:]
    else:
        cmd = args.rest
    if not cmd:
        print("❌ 命令为空", file=sys.stderr)
        sys.exit(99)

    log_path = Path(args.log)
    meta_path = log_path.with_suffix(log_path.suffix + ".watchdog.json")

    _write_event(meta_path, {
        "event": "watchdog_start",
        "ts": _ts(),
        "tag": args.tag,
        "cmd": cmd,
        "idle_timeout": args.idle_timeout,
        "absolute_timeout": args.absolute_timeout or None,
        "max_retries": args.max_retries,
    })

    final_rc = -1
    for attempt in range(1, args.max_retries + 2):
        print(f"\n=== watchdog 尝试 {attempt}/{args.max_retries + 1} ===")
        rc, reason = run_once(cmd, log_path, args.idle_timeout,
                              args.absolute_timeout or None)
        _write_event(meta_path, {
            "event": "attempt_end", "ts": _ts(), "attempt": attempt,
            "rc": rc, "reason": reason,
        })
        final_rc = rc
        if reason == "exited" and rc == 0:
            _write_event(meta_path, {"event": "success", "ts": _ts(), "attempt": attempt})
            print(f"✅ 成功（第 {attempt} 次尝试）")
            return 0
        if reason == "exited" and rc != 0:
            # 真的非 0 退出码，不是 idle 卡死：判定为"业务错误"，不重试
            print(f"⚠️  子进程返回非零 rc={rc}，停止重试")
            _write_event(meta_path, {"event": "fatal_nonzero_exit", "ts": _ts(), "rc": rc})
            return rc
        if attempt > args.max_retries:
            print(f"❌ 已达最大重试次数 {args.max_retries}，原因={reason}")
            _write_event(meta_path, {"event": "max_retries_reached", "ts": _ts(), "reason": reason})
            return final_rc
        print(f"⚠️  {reason}，{args.retry_sleep}s 后重试")
        time.sleep(args.retry_sleep)


if __name__ == "__main__":
    sys.exit(main())
