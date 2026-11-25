"""Run an installed file or mic pipeline and a dedicated Ollama under macOS Seatbelt."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener

import psutil


class Control(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"control":true}')

    def log_message(self, *args):
        pass


def observe_processes(root, role, samples):
    try:
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            with process.oneshot():
                rss = process.memory_info().rss
                row = samples.setdefault(process.pid, {
                    "pid": process.pid, "parent_pid": process.ppid(),
                    "process": process.name(), "tree": role, "max_sampled_rss_bytes": 0,
                })
                row["max_sampled_rss_bytes"] = max(row["max_sampled_rss_bytes"], rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue


def run_observed(command, env, work, server_pid, opener, url, report):
    samples, models = {}, {}
    server = psutil.Process(server_pid)
    available = psutil.virtual_memory().available
    report["system_available_before_bytes"] = available
    with (work / "app.stdout").open("w") as stdout, (work / "app.stderr").open("w") as stderr:
        with subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr) as launched:
            report["app_pid"] = launched.pid
            app = psutil.Process(launched.pid)
            while launched.poll() is None:
                observe_processes(server, "dedicated_ollama", samples)
                observe_processes(app, "speechflow", samples)
                available = min(available, psutil.virtual_memory().available)
                try:
                    with opener.open(url + "/api/ps", timeout=.5) as response:
                        for model in json.load(response).get("models", []):
                            digest = model["digest"]
                            models[digest] = {field: model.get(field) for field in
                                              ("name", "digest", "size", "size_vram", "context_length")}
                except (OSError, ValueError, KeyError):
                    pass
                try:
                    launched.wait(timeout=.75)
                except subprocess.TimeoutExpired:
                    continue
            code = launched.returncode
    report["system_min_sampled_available_bytes"] = available
    report["process_samples"] = list(samples.values())
    report["resident_model_samples"] = list(models.values())
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("file", "mic"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seconds", default=10, type=float)
    args = parser.parse_args()
    if args.mode == "file" and not args.audio:
        parser.error("File mode requires --audio")
    root = Path(__file__).resolve().parent
    cli = Path(sys.executable).parent / "speechflow"
    if not cli.is_file():
        parser.error("Run with the Python environment containing the installed speechflow CLI")
    args.out.mkdir(parents=True, exist_ok=True)
    # Use an isolated output subdirectory per invocation. A prior success cannot pass this run.
    work = args.out / f"{args.mode}-{time.time_ns()}"
    work.mkdir(mode=0o700)
    profile = root / "loopback.sb"
    probe = ["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable, "-I", str(root / "probe.py")]
    outside = socket.gethostbyname("huggingface.co")
    with socket.create_connection((outside, 443), timeout=10):
        pass
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as candidate:
        candidate.settimeout(3)
        ipv6_baseline = candidate.connect_ex(("2606:4700:4700::1111", 443, 0, 0))
    control = ThreadingHTTPServer(("127.0.0.1", 0), Control)
    thread = threading.Thread(target=control.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as lease:
        lease.bind(("127.0.0.1", 0))
        port = lease.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    settings_file = work / "settings.toml"
    settings_file.write_text(f'ollama_url = "{url}"\noffline = true\nbackend = "{args.backend}"\n')
    # The children need the model cache, PATH and a writable temporary directory.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL") if k in os.environ}
    env.update(OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_NO_CLOUD="1", OLLAMA_NOPRUNE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SPEECHFLOW_CONFIG=str(settings_file))
    if "OLLAMA_MODELS" in os.environ:
        env["OLLAMA_MODELS"] = os.environ["OLLAMA_MODELS"]
    server_command = [*probe, outside, f"http://127.0.0.1:{control.server_port}",
                      shutil.which("ollama") or "ollama", "serve"]
    call = ["run", str(args.audio.resolve())] if args.mode == "file" else ["mic", "--seconds", str(args.seconds)]
    app_command = [*probe, outside, url + "/api/version", str(cli), *call, "--json", "--offline",
                   "--out", str(work / "output"), "--database", str(work / "runs.sqlite3")]
    child_proof = work / "asr-network.json"
    app_env = {**env, "PYTHONPATH": os.pathsep.join([str(root / "instrument"), str(root)]),
               "SPEECHFLOW_TEST_CONTROL": json.dumps([str(child_proof.resolve()), outside, url + "/api/version"])}
    report = {"profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(), "outside": outside,
              "unsandboxed_tcp_443": True, "unsandboxed_ipv6_errno": ipv6_baseline,
              "mode": args.mode, "server_command": server_command,
              "app_command": app_command, "endpoint": url, "result": "BLOCKED"}
    opener = build_opener(ProxyHandler({}))
    try:
        with (work / "server-control.log").open("w") as log:
            server = subprocess.Popen(server_command, env=env, stdout=log, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
            try:
                for _ in range(150):
                    if server.poll() is not None:
                        raise RuntimeError(f"Dedicated Ollama exited {server.returncode}")
                    try:
                        with opener.open(url + "/api/version", timeout=.5) as response:
                            report["ollama_version"] = json.load(response)["version"]
                        break
                    except OSError:
                        time.sleep(.1)
                else:
                    raise RuntimeError("Dedicated Ollama did not become ready")
                report["server_control"] = json.loads((work / "server-control.log").read_text().splitlines()[0])
                if report["server_control"]["pid"] != server.pid:
                    raise RuntimeError("Server control PID does not match launched process")
                preflight = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), str(cli),
                                            "doctor", "--json", "--offline"], env=env, text=True,
                                           capture_output=True)
                report["preflight"] = json.loads(preflight.stdout)
                report["exit_code"] = run_observed(
                    app_command, app_env, work, server.pid, opener, url, report)
                report["result"] = "FAIL" if report["preflight"]["ready"] else "BLOCKED"
                report["app_control"] = json.loads((work / "app.stdout").read_text().splitlines()[0])
                if report["app_control"]["pid"] != report["app_pid"]:
                    raise RuntimeError("App control PID does not match the installed CLI")
                if child_proof.exists():
                    report["asr_child_control"] = json.loads(child_proof.read_text())
                    if report["asr_child_control"]["parent_pid"] != report["app_pid"]:
                        raise RuntimeError("ASR child control did not come from the installed CLI")
                manifest_path = work / "output/run.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    report["run_id"] = manifest["run_id"]
                    report["transcription_status"] = manifest["transcription_status"]
                    report["extraction_status"] = manifest["extraction_status"]
                    if (child_proof.exists() and report["exit_code"] == 0 and manifest["status"] == "ok" and
                            any(s["raw"].get("text") for s in manifest["segments"]) and
                            manifest["observations"] and manifest["attempts"] and
                            manifest["provenance"].get("extractor_digest")):
                        report["result"] = "PASS"
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=15)
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        report["error"] = str(exc)
    finally:
        control.shutdown()
        control.server_close()
        (work / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(work / "verification.json"), "status": report["result"]}))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
