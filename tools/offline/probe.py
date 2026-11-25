"""Verify positive and negative controls before exec in the same sandboxed process."""
import errno
import json
import os
import socket
import sys
import urllib.request

def check(ip, loopback):
    try:
        with socket.create_connection((ip, 443), timeout=5):
            raise RuntimeError("Unexpected external IPv4 connection")
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.EACCES):
            raise
        blocked = {"address": ip, "port": 443, "errno": exc.errno, "message": str(exc)}
    ipv6 = "2606:4700:4700::1111"
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as candidate:
        candidate.settimeout(3)
        v6_errno = candidate.connect_ex((ipv6, 443, 0, 0))
    if v6_errno == 0:
        raise RuntimeError("Unexpected external IPv6 connection")
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(loopback, timeout=3) as response:
        loop = {"url": loopback, "status": response.status, "body": json.load(response)}
    return {"pid": os.getpid(), "denied": blocked, "ipv6": {"address": ipv6, "errno": v6_errno},
            "allowed": loop}


if __name__ == "__main__":
    ip, loopback, *command = sys.argv[1:]
    print(json.dumps({**check(ip, loopback), "exec": command[:2]}), flush=True)
    if command:
        os.execvp(command[0], command)
