"""在 import datasets / huggingface_hub / httpx 之前调用，修复 TLS CA bundle。

常见报错：
  Could not find a suitable TLS CA certificate bundle, invalid path:
  .../site-packages/certifi/cacert.pem

原因：certifi 不完整、venv 内 cacert.pem 损坏/空文件；或 **项目路径含非 ASCII（如中文）**
时，部分 OpenSSL/urllib3 组合对 certifi 返回的路径处理异常。

策略：
  - **macOS (Darwin)**：若存在 `/etc/ssl/cert.pem`，**优先强制使用**（覆盖 certifi），
    避免中文路径 + 坏 certifi 双重问题。
  - 其他系统：certifi 可用且文件足够大则用 certifi，否则回退系统 CA。
"""
from __future__ import annotations

import os
import sys


def apply_hf_ssl_fix(verbose: bool = True) -> str | None:
    """设置 SSL 相关环境变量；返回实际使用的 CA 文件路径（若成功）。"""
    def _ok_pem(p: str | None) -> bool:
        if not p or not os.path.isfile(p):
            return False
        try:
            return os.path.getsize(p) >= 4096  # 真 cacert 至少数 KB；空/坏文件排除
        except OSError:
            return False

    # 用户已显式配置且有效 → 尊重
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        v = os.environ.get(key)
        if _ok_pem(v):
            return v

    # macOS：系统根证书链最稳（避免中文项目路径 + certifi 路径触发 invalid path）
    if sys.platform == "darwin":
        for c in ("/etc/ssl/cert.pem", "/opt/homebrew/etc/openssl@3/cert.pem",
                  "/usr/local/etc/openssl@3/cert.pem"):
            if _ok_pem(c):
                os.environ["SSL_CERT_FILE"] = c
                os.environ["REQUESTS_CA_BUNDLE"] = c
                os.environ["CURL_CA_BUNDLE"] = c
                if verbose:
                    print(f"ℹ️  macOS：使用系统 CA 链（覆盖 certifi）→ {c}")
                return c

    try:
        import certifi

        p = certifi.where()
        if _ok_pem(p):
            os.environ.setdefault("SSL_CERT_FILE", p)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", p)
            os.environ.setdefault("CURL_CA_BUNDLE", p)
            return p
    except Exception:
        pass

    candidates = (
        "/etc/ssl/cert.pem",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/local/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
    )
    for c in candidates:
        if _ok_pem(c):
            os.environ["SSL_CERT_FILE"] = c
            os.environ["REQUESTS_CA_BUNDLE"] = c
            os.environ["CURL_CA_BUNDLE"] = c
            if verbose:
                print(f"⚠️  certifi CA 无效，已回退系统 CA：{c}")
            return c
    if verbose:
        print("❌ 未找到可用 CA 文件。请执行：python -m pip install --force-reinstall certifi")
    return None
