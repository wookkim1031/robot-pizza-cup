"""Web UI for the pizza runs: start a fixed challenge sequence (run_challenge.py) or the Claude agent
(run_agent.py), watch the log live, stop it. One run at a time; every run's log is saved to logs/runs/.

    uv run python scripts/web_ui.py            # then open http://127.0.0.1:8890

Stop sends Ctrl-C (SIGINT), so the run prints its summary; pressing it again kills the run.
Each run also gets a checker log dir (logs/runs/<run>.checker/, see pizza/check_log.py); the page shows its
events: with failure checker v2 the located pizza and sector boxes at the start, and every checked step.
The Anthropic API key can be saved from the page (stored in ~/.config/pizza-ui/, see pizza/apikey.py)
or exported as ANTHROPIC_API_KEY before starting this server. Local Ollama models on gpu1 show up in the
model list as "local" and need no key: the page shows whether gpu1 is connected, "Connect now" opens the tunnel
and loads the model, and Run does the same by itself if needed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_agent  # noqa: E402
import run_challenge  # noqa: E402
from pizza import apikey  # noqa: E402
from pizza.recipe import CHALLENGES  # noqa: E402
from pizza.robot import DEFAULT_CHECKPOINT  # noqa: E402
from pizza.skills import DriveConfig  # noqa: E402
from pizza.tasks import INGREDIENTS, TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "runs"
DEFAULT_MODEL = "claude-opus-5-5"
STATIC_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1", "claude-opus-5"]
# Options that have their own control on the page; the rest are listed as extra args.
UI_DESTS = {"help", "challenge", "text", "recipe", "checkpoint", "check", "drive", "sim", "dry_run", "time_limit",
            "model", "effort", "server", "confirm", "failure_checker", "checker_log"}
CHECKER_IMG = re.compile(r"^\d{3}-\w+\.png$")
# Standard wordings, as quick fills for the agent prompt.
STANDARD_PROMPTS = {
    "Ch1 cheese": "Spread the cheese relatively evenly over the pizza.",
    "Ch1 pepperoni": "Place 6 pepperoni pieces on the pizza and spread them over the pizza relatively evenly.",
    "Ch2 / Ch3A": "Make a pizza: spread the cheese relatively evenly over the pizza, then place 6 pepperoni "
                  "pieces spread relatively evenly over the pizza.",
}


def checkpoints() -> list[str]:
    """Policy checkpoint dirs (relative to the project), newest first."""
    found = [p.parent for p in (ROOT / "checkpoints").glob("*/model.safetensors")]
    found += [p.parent for p in (ROOT / "checkpoints").glob("*/final/model.safetensors")]
    return [str(p.relative_to(ROOT)) for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]


def arg_reference(parser: argparse.ArgumentParser) -> list[dict]:
    """The script's own options minus the ones the page has controls for."""
    out = []
    for a in parser._actions:
        if not a.option_strings or a.dest in UI_DESTS:
            continue
        flag = max(a.option_strings, key=len)
        value = "" if a.nargs == 0 else (a.metavar or a.dest.upper())
        default = None if a.default in (None, False, [], argparse.SUPPRESS) else a.default
        out.append({"flag": flag, "value": value, "default": str(default) if default is not None else "",
                    "help": (a.help or "").replace("%%", "%")})
    return out


def drive_reference() -> list[dict]:
    """DriveConfig fields with their defaults and the comment next to each (pizza/skills.py)."""
    src = inspect.getsource(DriveConfig)
    return [{"key": m[1], "default": m[2].strip(), "help": m[3]}
            for m in re.finditer(r"^    (\w+): [\w .|]+ = ([^#\n]+?)\s*# (.+)$", src, re.M)]


_models_cache: tuple[float, list[str], str] = (0.0, [], "")


def ollama_models() -> list[str]:
    """Models on the Ollama server as ollama:<name>; just the default one if the server is not reachable."""
    try:
        names = [run_agent.OLLAMA_PREFIX + m["name"]
                 for m in run_agent.ollama_api(run_agent.OLLAMA_URL, "/api/tags", timeout=2).get("models", [])]
    except (OSError, ValueError, KeyError):
        names = []
    return sorted(set(names) | {run_agent.LOCAL_MODEL}, key=lambda m: (m != run_agent.LOCAL_MODEL, m))


def list_models() -> tuple[list[str], dict[str, str], str]:
    """Claude models plus the local Ollama models, and display labels for the local ones."""
    ids, source = claude_models()
    local = ollama_models()
    labels = {m: f"{m.removeprefix(run_agent.OLLAMA_PREFIX)} · local on gpu1 (no API key)" for m in local}
    return ids[:1] + local + ids[1:], labels, source


class Ollama:
    """Connect-and-load of the gpu1 model in the background, so the page can show progress."""

    def __init__(self):
        self.lock = threading.Lock()
        self.busy, self.msg = False, ""

    def status(self) -> dict:
        try:
            loaded = [run_agent.OLLAMA_PREFIX + m["name"]
                      for m in run_agent.ollama_api(run_agent.OLLAMA_URL, "/api/ps", timeout=2).get("models", [])]
            up = True
        except (OSError, ValueError, KeyError):
            loaded, up = [], False
        return {"up": up, "loaded": loaded, "busy": self.busy, "msg": self.msg}

    def connect(self, model: str) -> None:
        with self.lock:
            if self.busy:
                return
            self.busy, self.msg = True, "connecting to gpu1 and loading the model (up to ~2 min the first time) ..."
        threading.Thread(target=self._connect, args=(model,), daemon=True).start()

    def _connect(self, model: str) -> None:
        try:
            run_agent.ollama_warmup(run_agent.OLLAMA_URL, model.removeprefix(run_agent.OLLAMA_PREFIX))
            self.msg = ""
        except Exception as e:  # shown on the page
            self.msg = f"failed: {e}"
        finally:
            self.busy = False


OLLAMA = Ollama()


def claude_models() -> tuple[list[str], str]:
    """Claude models from the Models API (ones that take images and effort), cached 5 min."""
    global _models_cache
    key = apikey.load()
    if not key:
        return STATIC_MODELS, "built-in list (no API key yet)"
    if time.time() - _models_cache[0] < 300 and _models_cache[1]:
        return _models_cache[1], _models_cache[2]
    try:
        ids = []
        for m in anthropic.Anthropic(api_key=key, timeout=15).models.list():
            caps = getattr(m, "capabilities", None)
            try:
                if caps and not (caps["image_input"]["supported"] and caps["effort"]["low"]["supported"]):
                    continue
            except (KeyError, TypeError):
                pass
            ids.append(m.id)
    except anthropic.AuthenticationError:
        return STATIC_MODELS, "built-in list: the API key was rejected"
    except anthropic.APIError as e:
        return STATIC_MODELS, f"built-in list: Models API failed ({type(e).__name__})"
    ids.sort(key=lambda i: (i != DEFAULT_MODEL, i))
    _models_cache = (time.time(), ids, "from the Models API")
    return ids, _models_cache[2]


class Runner:
    """The one child process, its merged output, and a log file copy."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.cmd: list[str] = []
        self.started = 0.0
        self.code: int | None = None
        self.stops = 0
        self.log_path: Path | None = None
        self.checker_dir: Path | None = None

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], label: str, checker_log: bool = False) -> None:
        with self.lock:
            if self.busy():
                raise RuntimeError("a run is already in progress")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.log"
            self.checker_dir = self.log_path.with_suffix(".checker") if checker_log else None
            if self.checker_dir:
                cmd = cmd + ["--checker-log", str(self.checker_dir.relative_to(ROOT))]
            self.lines, self.cmd, self.code, self.stops, self.started = [], cmd, None, 0, time.time()
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc, self.log_path), daemon=True).start()

    def _pump(self, proc: subprocess.Popen, log_path: Path) -> None:
        with open(log_path, "w") as log:
            log.write(f"$ {shlex.join(self.cmd)}\n")
            for line in proc.stdout:
                self.lines.append(line.rstrip("\n"))
                log.write(line)
                log.flush()
            self.code = proc.wait()
            log.write(f"[exit {self.code}]\n")

    def stop(self) -> str:
        if not self.busy():
            return "not running"
        self.stops += 1
        if self.stops == 1:
            self.proc.send_signal(signal.SIGINT)
            return "sent Ctrl-C (press Stop again to kill)"
        self.proc.kill()
        return "killed"

    def state(self, since: int) -> dict:
        return {"running": self.busy(), "code": self.code, "cmd": shlex.join(self.cmd) if self.cmd else "",
                "elapsed": round(time.time() - self.started) if self.started else 0,
                "log": str(self.log_path.relative_to(ROOT)) if self.log_path else "",
                "lines": self.lines[since:], "next": len(self.lines)}

    def checker(self, since: int) -> dict:
        """Checker events of the current run from line `since` on."""
        path = self.checker_dir / "events.jsonl" if self.checker_dir else None
        lines = path.read_text().splitlines() if path and path.exists() else []
        return {"dir": str(self.checker_dir.relative_to(ROOT)) if self.checker_dir else "",
                "events": [json.loads(x) for x in lines[since:]], "next": len(lines)}


RUNNER = Runner()


def build_command(req: dict) -> tuple[list[str], str]:
    """Run request from the page -> argv for run_challenge.py or run_agent.py."""
    mode = req.get("mode")
    common = []
    if req.get("checkpoint"):
        common += ["--checkpoint", req["checkpoint"]]
    if req.get("check") in ("on", "off"):
        common += ["--check" if req["check"] == "on" else "--no-check"]
    if req.get("failure_checker") in ("v1", "v2"):
        common += ["--failure-checker", req["failure_checker"]]
    if str(req.get("time_limit") or "").strip():
        common += ["--time-limit", str(float(req["time_limit"]))]
    if req.get("sim"):
        common += ["--sim"]
    extra = shlex.split(req.get("extra") or "")
    py = [sys.executable, "-u"]
    if mode == "challenge":
        ch = req.get("challenge")
        if ch == "one":
            if req.get("ingredient") not in INGREDIENTS or req.get("target") not in TARGETS:
                raise ValueError("pick an ingredient and a target")
            args = ["one", req["ingredient"], req["target"]]
        elif ch in CHALLENGES:
            args = [ch]
            if CHALLENGES[ch].steps is None:
                if not (req.get("recipe") or "").strip():
                    raise ValueError(f"{ch} needs the recipe text")
                args.append(req["recipe"].strip())
        else:
            raise ValueError(f"unknown challenge {ch!r}")
        if req.get("dry_run"):
            common += ["--dry-run"]
        return py + ["scripts/run_challenge.py", *args, *common, *extra], ch
    if mode == "agent":
        prompt = (req.get("recipe") or "").strip()
        if not prompt:
            raise ValueError("write the prompt for the agent")
        if not apikey.load() and not run_agent.is_local(req.get("model") or DEFAULT_MODEL):
            raise ValueError("no Anthropic API key: save one in the API key box first (sim runs use Claude too)")
        args = [prompt, "--model", req.get("model") or DEFAULT_MODEL, "--effort", req.get("effort") or "low"]
        return py + ["scripts/run_agent.py", *args, *common, *extra], "agent"
    raise ValueError(f"unknown mode {mode!r}")


def key_status() -> dict:
    key = apikey.load()
    source = "environment (ANTHROPIC_API_KEY)" if os.environ.get("ANTHROPIC_API_KEY") else \
        (f"saved in {apikey.KEY_FILE}" if key else "")
    return {"set": bool(key), "masked": apikey.masked(key), "source": source}


def config() -> dict:
    budgets = {k: (c.time_total or sum(c.time.values())) for k, c in CHALLENGES.items()}
    return {"checkpoints": checkpoints(), "default_checkpoint": str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
            "challenges": {k: c.title for k, c in CHALLENGES.items()},
            "challenge_check": {k: c.check for k, c in CHALLENGES.items()},
            "budgets": budgets, "ingredients": INGREDIENTS, "targets": TARGETS,
            "args": {"challenge": arg_reference(run_challenge.build_parser()),
                     "agent": arg_reference(run_agent.build_parser())},
            "drive": drive_reference(), "prompts": STANDARD_PROMPTS, "key": key_status()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet: the page polls twice a second
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/config":
            self._json(config())
        elif self.path == "/models":
            ids, labels, source = list_models()
            self._json({"models": ids, "labels": labels, "source": source, "default": DEFAULT_MODEL})
        elif self.path == "/ollama":
            self._json(OLLAMA.status())
        elif self.path.startswith("/state"):
            self._json(RUNNER.state(int(self.path.partition("since=")[2] or 0)))
        elif self.path.startswith("/checker/img/"):
            name = self.path.rpartition("/")[2]
            f = RUNNER.checker_dir / name if RUNNER.checker_dir and CHECKER_IMG.match(name) else None
            if not f or not f.exists():
                return self._json({"error": "not found"}, 404)
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/checker"):
            self._json(RUNNER.checker(int(self.path.partition("since=")[2] or 0)))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        req = self._body()
        if self.path == "/run":
            try:
                cmd, label = build_command(req)
                RUNNER.start(cmd, label, checker_log=True)
            except (ValueError, RuntimeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})
        elif self.path == "/ollama/connect":
            model = req.get("model") or run_agent.LOCAL_MODEL
            if not run_agent.is_local(model):
                return self._json({"error": f"{model} is not a local model"}, 400)
            OLLAMA.connect(model)
            self._json(OLLAMA.status())
        elif self.path == "/stop":
            self._json({"msg": RUNNER.stop()})
        elif self.path == "/home":
            try:
                RUNNER.start([sys.executable, "-u", "scripts/go_home.py", "--seconds", str(float(req.get("seconds") or 6))],
                             "home")
            except (ValueError, RuntimeError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True})
        elif self.path == "/key":
            global _models_cache
            key = (req.get("key") or "").strip()
            if not key:
                apikey.clear()
            elif not key.startswith("sk-ant-"):
                return self._json({"error": "that does not look like an Anthropic API key (sk-ant-...)"}, 400)
            else:
                apikey.save(key)
            _models_cache = (0.0, [], "")
            self._json(key_status())
        else:
            self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pizza Runner</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#dddbd5;--acc:#c2410c;--ok:#15803d;--bad:#b91c1c;--log:#141413;--logink:#e8e6df;--chip:#f0eee9}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--panel:#232321;--ink:#ecebe6;--mute:#a09f99;--line:#3a3935;--acc:#f97316;--ok:#4ade80;--bad:#f87171;--log:#0f0f0e;--chip:#2e2e2b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px;display:grid;grid-template-columns:420px 1fr;gap:16px}
@media (max-width:860px){main{grid-template-columns:1fr}}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);font-size:13px;margin-bottom:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:12px}
.tabs{display:flex;gap:4px;margin-bottom:6px;background:var(--bg);padding:4px;border-radius:8px}
.tabs button{flex:1;border:0;background:none;padding:8px;border-radius:6px;font-weight:600;color:var(--mute);cursor:pointer}
.tabs button.on{background:var(--panel);color:var(--ink);box-shadow:0 1px 2px #0002}
label{display:block;font-size:12px;font-weight:600;color:var(--mute);margin:12px 0 4px;text-transform:uppercase;letter-spacing:.03em}
select,input,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:inherit}
textarea{min-height:120px;resize:vertical}.row{display:flex;gap:8px}.row>*{flex:1;min-width:0}
.checks{display:flex;gap:16px;margin-top:12px}.checks label{display:flex;gap:6px;align-items:center;margin:0;text-transform:none;font-size:14px;color:var(--ink);font-weight:400}
.checks input{width:auto}
.btns{display:flex;gap:8px;margin-top:16px}.btns button{flex:1;padding:11px;border:0;border-radius:7px;font-weight:700;font-size:15px;cursor:pointer}
#run{background:var(--acc);color:#fff}#stop{background:var(--bad);color:#fff}#home{background:var(--chip);color:var(--ink);border:1px solid var(--line)!important;flex:0 0 110px}button:disabled{opacity:.4;cursor:default}
.hint{font-size:12px;color:var(--mute);margin-top:5px}.warn{color:var(--bad)}.good{color:var(--ok)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.chip{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:99px;padding:2px 9px;font-size:12px;cursor:pointer}
.small{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--chip);color:var(--ink);cursor:pointer;font-weight:600;flex:0 0 auto}
.status{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.pill{padding:3px 10px;border-radius:99px;font-size:13px;font-weight:700;background:var(--line)}
.pill.run{background:var(--acc);color:#fff}.pill.ok{background:var(--ok);color:#fff}.pill.bad{background:var(--bad);color:#fff}
#cmd{font:12px ui-monospace,monospace;color:var(--mute);word-break:break-all;margin-bottom:6px}
#log{background:var(--log);color:var(--logink);font:12.5px/1.5 ui-monospace,monospace;border-radius:8px;padding:12px;height:calc(100vh - 160px);min-height:320px;overflow:auto;white-space:pre-wrap;word-break:break-word}
.l-claude{color:#fdba74}.l-step{color:#93c5fd;font-weight:700}.l-bad{color:#fca5a5}.l-good{color:#86efac}
details{margin-top:10px}summary{cursor:pointer;font-size:13px;font-weight:600}
.ref{margin-top:6px;font-size:12px}.ref>div{border-top:1px solid var(--line);padding:5px 2px}
.ref code{font:12px ui-monospace,monospace;cursor:pointer;color:var(--acc);word-break:break-all}.ref code:hover{text-decoration:underline}
.ref .d{color:var(--mute);margin-left:6px}.ref .h{color:var(--ink);margin-top:1px}
.local{margin-top:10px;padding:10px;border:1px dashed var(--line);border-radius:8px}
.hidden{display:none}
#ck{margin-bottom:10px}#ck .card{margin-bottom:8px;padding:12px}
.ck-start{display:flex;gap:12px;align-items:flex-start}.ck-start a{flex:0 0 min(240px,45%)}.ck-start img{width:100%;display:block;border-radius:6px}
.ck-steps{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
.ck-step{flex:0 0 180px;border:1px solid var(--line);border-radius:8px;padding:6px;font-size:12px;background:var(--panel)}
.ck-step img{width:100%;border-radius:4px;display:block;margin-bottom:4px}
.ck-step b.ok{color:var(--ok)}.ck-step b.bad{color:var(--bad)}.ck-step .m{color:var(--mute)}
.ck-title{font-size:12px;font-weight:700;color:var(--mute);text-transform:uppercase;letter-spacing:.03em;margin-bottom:6px}
</style></head><body><main>
<section>
  <h1>Pizza Runner</h1><div class="sub">Learned SmolVLA skills · fixed challenge sequences or the Claude agent</div>
  <div class="tabs"><button id="t-challenge" class="on">Challenge (fixed sequence)</button><button id="t-agent">Agent (Claude / local)</button></div>
  <div class="card">
    <div id="challenge-box">
      <label>Challenge</label><select id="challenge"></select>
      <div id="one-box" class="row hidden">
        <div><label>Ingredient</label><select id="ingredient"></select></div>
        <div><label>Target</label><select id="target"></select></div>
      </div>
    </div>
    <div id="recipe-box">
      <label id="recipe-label">Prompt</label>
      <textarea id="recipe"></textarea>
      <div class="chips" id="prompt-chips"></div>
      <div class="hint" id="recipe-hint"></div>
    </div>
    <div class="row">
      <div><label>Time limit (s)</label><input id="time_limit" type="number" min="0">
        <div class="chips"><button class="chip" data-t="180">180 · Ch1</button><button class="chip" data-t="480">480 · Ch2/3</button></div></div>
      <div id="model-box" class="hidden"><label>Model</label><select id="model"></select><div class="hint" id="model-src"></div></div>
    </div>
    <div id="local-box" class="hidden local">
      <div class="row" style="align-items:center"><div class="hint" id="local-status" style="margin:0"></div><button class="small" id="local-connect">Connect now</button></div>
      <div class="hint">Runs on gpu1 through an SSH tunnel (needs internet, no API key). Run connects by itself; Connect now just preloads the model so the first decision is not slow.</div>
    </div>
    <div class="hint" id="time-hint"></div>
    <div class="row">
      <div><label>Change checker</label><select id="check"></select></div>
      <div><label>Checker version</label><select id="failure_checker"><option value="v2">v2 · boxes follow the pizza</option><option value="v1">v1 · fixed boxes</option></select></div>
      <div id="effort-box" class="hidden"><label>Effort</label><select id="effort"><option>low</option><option>medium</option><option>high</option></select></div>
    </div>
    <div class="hint">Checker: after each pepperoni step it compares the scene before and after to see if a slice landed in the target quadrant (and retries if not). Cheese is never checked. v1 uses fixed quadrant boxes, so it only works with the pizza in its usual spot. v2 finds the pizza once at the start (arm at home, before any pepperoni) and moves the boxes with it; up to ~40 px off (a ch3 shift is 10-30 px), beyond that or if not found it falls back to v1. A dry run with v2 locates the pizza on the live image without moving the arm.</div>
    <label>Checkpoint</label><select id="checkpoint"></select>
    <label>Extra args</label><input id="extra" placeholder="click an option below to add it">
    <details><summary>Available extra args (click one to add it)</summary>
      <div class="ref" id="args-table"></div>
      <div class="hint" style="margin-top:10px"><b>--drive KEY=VALUE</b> (step execution, repeatable):</div>
      <div class="ref" id="drive-table"></div>
    </details>
    <div class="checks"><label><input type="checkbox" id="sim">Sim (no robot)</label>
      <label id="dry-wrap"><input type="checkbox" id="dry_run">Dry run (plan only)</label></div>
    <div class="btns"><button id="run">Run</button><button id="stop" disabled>Stop</button><button id="home" title="Slow 6 s move to the home pose, then open the gripper">Go home</button></div>
    <div class="hint">Before a real run: station up, arm parked at home, gripper open. Teleop is stopped automatically.</div>
  </div>
  <div class="card" id="key-card">
    <label style="margin-top:0">Anthropic API key (agent)</label>
    <div class="row"><input id="key" type="password" placeholder="sk-ant-..." autocomplete="off"><button class="small" id="key-save">Save</button><button class="small" id="key-clear">Clear</button></div>
    <div class="hint" id="key-status"></div>
  </div>
</section>
<section>
  <div class="status"><span class="pill" id="pill">idle</span><span id="elapsed"></span><span id="logpath" class="hint"></span></div>
  <div id="cmd"></div>
  <div id="ck" class="hidden">
    <div class="card" id="ck-start-card"><div class="ck-title">Failure checker · start</div><div class="ck-start"><a id="ck-start-link" target="_blank"><img id="ck-start-img" alt="located pizza and sector boxes"></a><div id="ck-start-text" class="hint" style="margin:0"></div></div></div>
    <div class="card hidden" id="ck-steps-card"><div class="ck-title">Checked steps (newest first)</div><div class="ck-steps" id="ck-steps"></div></div>
  </div>
  <div id="log"></div>
</section>
</main><script>
const $=id=>document.getElementById(id);let cfg,mode="challenge",next=0,wasRunning=false,agentTime=480;const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function opts(sel,items,labels){sel.innerHTML=items.map(v=>`<option value="${esc(v)}">${esc(labels&&labels[v]||v)}</option>`).join("")}
function addExtra(s){const e=$("extra");e.value=(e.value.trim()+" "+s).trim();e.focus()}
function renderArgs(){
 $("args-table").innerHTML=cfg.args[mode].map(a=>`<div><code data-a="${esc(a.flag+(a.value?" "+(a.default||a.value):""))}">${esc(a.flag)}${a.value?" "+esc(a.value):""}</code><span class="d">${a.default?"default "+esc(a.default):""}</span><div class="h">${esc(a.help)}</div></div>`).join("");
 $("drive-table").innerHTML=cfg.drive.map(d=>`<div><code data-a="--drive ${esc(d.key)}=${esc(d.default)}">${esc(d.key)}</code><span class="d">default ${esc(d.default)}</span><div class="h">${esc(d.help)}</div></div>`).join("");
 document.querySelectorAll(".ref code").forEach(c=>c.onclick=()=>addExtra(c.dataset.a))}
function setMode(m){mode=m;localUpdate();$("t-challenge").classList.toggle("on",m=="challenge");$("t-agent").classList.toggle("on",m=="agent");
 const agent=m=="agent";$("challenge-box").classList.toggle("hidden",agent);
 ["model-box","effort-box"].forEach(id=>$(id).classList.toggle("hidden",!agent));$("dry-wrap").classList.toggle("hidden",agent);
 $("extra").value="";renderArgs();update();if(agent)loadModels()}
function update(){const ch=$("challenge").value,agent=mode=="agent";
 $("one-box").classList.toggle("hidden",agent||ch!="one");
 $("recipe-box").classList.toggle("hidden",!agent&&ch!="ch3b");
 $("prompt-chips").classList.toggle("hidden",!agent);
 $("recipe-label").textContent=agent?"Prompt (exactly as the judges give it)":"Recipe text (parsed by rules)";
 $("recipe").placeholder=agent?"e.g. Place 4 pepperoni slices on the right half of the pizza, then spread cheese over the whole pizza.":"e.g. 4 pepperoni on the right half, then cheese over the whole pizza";
 $("recipe-hint").textContent=agent?"Claude works out the task from this text: one topping (Ch1), a full recipe, counts, regions, order.":"Rules turn the text into steps: stage order, counts, halves/quadrants, center. Tick Dry run to check the plan first.";
 if(agent){opts($("check"),["on","off"],{on:"On: pizza in its usual spot",off:"Off: pizza moved (Ch3)"});
  $("time_limit").value=agentTime;
  $("time-hint").textContent="The agent sees the time used/left after every step and the script stops new steps at the limit."}
 else{const on=cfg.challenge_check[ch]??true;opts($("check"),["","on","off"],{"":`Challenge default (${on?"on":"off"} for this one)`,on:"On",off:"Off"});
  const b=cfg.budgets[ch];$("time_limit").value=b||"";
  $("time-hint").textContent=ch=="one"?"Empty = no limit.":`Default ${b}s (official limit minus ~10 s). Split over the ingredients; a step that would not fit in what is left is skipped. 0 = no limit.`}}
const store={get:k=>{try{return localStorage.getItem(k)}catch(e){return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
const isLocal=m=>(m||"").startsWith("ollama:");
async function loadModels(){const sel=$("model"),cur=sel.value||store.get("model");$("model-src").textContent="loading models…";
 try{const j=await (await fetch("/models")).json();opts(sel,j.models,j.labels);sel.value=cur&&j.models.includes(cur)?cur:j.default;$("model-src").textContent=j.source}catch(e){$("model-src").textContent="could not load models"}
 localUpdate()}
let ltimer;async function localUpdate(){clearTimeout(ltimer);const m=$("model").value,show=mode=="agent"&&isLocal(m);
 $("local-box").classList.toggle("hidden",!show);if(!show)return;
 let s;try{s=await (await fetch("/ollama")).json()}catch(e){return}
 const ready=s.loaded.includes(m),st=$("local-status");
 st.innerHTML=s.busy?`<span>⏳ ${esc(s.msg)}</span>`:ready?`<span class="good">● gpu1 connected, model loaded: ready</span>`:
  s.up?`<span>● gpu1 connected, model not loaded yet (Run loads it, ~75 s)</span>`:
  `<span class="warn">● gpu1 not connected</span> (Run connects by itself)`+(s.msg?` <span class="warn">${esc(s.msg)}</span>`:"");
 $("local-connect").disabled=s.busy||ready;$("local-connect").textContent=s.busy?"Connecting…":"Connect now";
 ltimer=setTimeout(localUpdate,s.busy?1500:5000)}
function showKey(k){$("key-status").innerHTML=k.set?`<span class="good">Key set</span> (${esc(k.masked)}) · ${esc(k.source)}`:`<span class="warn">No key.</span> Paste it here and Save; it is stored only on this laptop (~/.config/pizza-ui/, not in the repo).`}
async function init(){cfg=await (await fetch("/config")).json();
 opts($("challenge"),[...Object.keys(cfg.challenges),"one"],{...cfg.challenges,one:"Single step (test)"});
 opts($("ingredient"),cfg.ingredients);opts($("target"),cfg.targets);
 opts($("checkpoint"),cfg.checkpoints);$("checkpoint").value=cfg.default_checkpoint;
 $("prompt-chips").innerHTML=Object.entries(cfg.prompts).map(([k,v])=>`<button class="chip" data-p="${esc(v)}">${esc(k)}</button>`).join("");
 document.querySelectorAll("[data-p]").forEach(b=>b.onclick=()=>{$("recipe").value=b.dataset.p});
 document.querySelectorAll("[data-t]").forEach(b=>b.onclick=()=>{$("time_limit").value=b.dataset.t;if(mode=="agent")agentTime=b.dataset.t});
 $("t-challenge").onclick=()=>setMode("challenge");$("t-agent").onclick=()=>setMode("agent");$("challenge").onchange=update;
 showKey(cfg.key);setMode("challenge");poll()}
async function saveKey(key){const r=await fetch("/key",{method:"POST",body:JSON.stringify({key})});const j=await r.json();
 if(!r.ok){alert(j.error);return}$("key").value="";showKey(j);if(mode=="agent")loadModels()}
$("time_limit").oninput=()=>{if(mode=="agent")agentTime=$("time_limit").value};
$("model").onchange=()=>{store.set("model",$("model").value);localUpdate()};
$("local-connect").onclick=async()=>{await fetch("/ollama/connect",{method:"POST",body:JSON.stringify({model:$("model").value})});localUpdate()};
$("key-save").onclick=()=>saveKey($("key").value);$("key-clear").onclick=()=>saveKey("");
$("run").onclick=async()=>{const body={mode};for(const k of["challenge","ingredient","target","recipe","time_limit","model","effort","checkpoint","check","failure_checker","extra"])body[k]=$(k).value;
 body.sim=$("sim").checked;body.dry_run=$("dry_run").checked&&mode=="challenge";
 const r=await fetch("/run",{method:"POST",body:JSON.stringify(body)});const j=await r.json();
 if(!r.ok){alert(j.error);return}$("log").innerHTML="";next=0;ckReset();poll()};
$("home").onclick=async()=>{if(!confirm("Move the arm slowly to the home pose and open the gripper?"))return;
 const r=await fetch("/home",{method:"POST",body:"{}"});const j=await r.json();if(!r.ok){alert(j.error);return}$("log").innerHTML="";next=0;ckReset();poll()};
$("stop").onclick=async()=>{const j=await (await fetch("/stop",{method:"POST"})).json();addLine("[ui] "+j.msg)};
function cls(l){if(l.startsWith("claude")||l.startsWith("local"))return"l-claude";if(/^\[\d+(\/\d+)?\]/.test(l))return"l-step";
 if(/TIMEOUT|NOT added|failed|Error|Traceback|refused|gave up|SKIPPED|no Anthropic/.test(l))return"l-bad";if(/ADDED|finished|summary/.test(l))return"l-good";return""}
function addLine(l){const d=document.createElement("div");d.textContent=l;d.className=cls(l);const log=$("log");
 const atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-30;log.appendChild(d);if(atEnd)log.scrollTop=log.scrollHeight}
let cnext=0,ckChecks={},ckDecisions={};
function ckReset(){cnext=0;ckChecks={};ckDecisions={};$("ck-start-img").removeAttribute("src");$("ck").classList.add("hidden");$("ck-steps-card").classList.add("hidden");$("ck-steps").innerHTML="";$("ck-start-card").classList.remove("hidden")}
const pct=v=>v==null?"–":Number(v).toFixed(2);
function ckStart(e){$("ck").classList.remove("hidden");const img=$("ck-start-img"),u="/checker/img/"+e.image;img.src=u;$("ck-start-link").href=u;
 const st=e.fallback?`<span class="warn">FALLBACK to v1 boxes</span>`:e.boxes_source=="dynamic"?`<span class="good">found · boxes moved</span>`:`<span class="good">found · fixed boxes</span>`;
 $("ck-start-text").innerHTML=`<div style="font-size:14px;margin-bottom:4px">${st}</div>`+
  (e.found?`pizza center ${e.center.map(v=>v.toFixed(1)).join(", ")} · <b>${e.offset_px} px</b> from the reference`+(e.shift_px?` (dx ${e.shift_px[0]}, dy ${e.shift_px[1]})`:""):"pizza not found")+
  (e.reason?`<br>${esc(e.reason)}`:"")+`<br>radius ${e.radius} px · method ${esc(e.method)} · ${e.ms} ms`}
function ckDraw(seq){const e=ckChecks[seq],d=ckDecisions[seq]||{};let el=document.getElementById("ck-"+seq);
 if(!el){el=document.createElement("div");el.className="ck-step";el.id="ck-"+seq;$("ck-steps").prepend(el)}
 const verdict=!e.supported?`<b>not checked</b> <span class="m">${esc(e.reason||"")}</span>`:`<b class="${e.ok?"ok":"bad"}">${e.ok?"OK":"MISSING"}</b> p=${pct(e.p)} <span class="m">(thr ${e.threshold})</span>`;
 const els=Object.entries(e.elsewhere||{}).map(([k,v])=>`${k} ${pct(v)}`).join(", ");
 const dec=d.attempt?`attempt ${d.attempt}/${d.attempts_max} · `+(d.retry?`<b class="bad">retry</b>`:d.gave_up?`<b class="bad">gave up</b>`:"no retry"):"";
 el.innerHTML=(e.image?`<a href="/checker/img/${e.image}" target="_blank"><img src="/checker/img/${e.image}"></a>`:"")+
  `<div>${d.step?`<span class="m">[${d.step}/${d.steps}]</span> `:""}<b>${esc(e.topping)} ${esc(e.target)}</b> · ${esc(e.version)}</div><div>${verdict}</div>`+
  (els?`<div class="warn">also in: ${esc(els)}</div>`:"")+(dec?`<div>${dec}</div>`:"")+`<div class="m">boxes: ${esc(e.boxes_source||"")}</div>`}
async function ckPoll(){try{const c=await (await fetch("/checker?since="+cnext)).json();cnext=c.next;
 for(const e of c.events){if(e.kind=="start")ckStart(e);
  else if(e.kind=="check"){ckChecks[e.seq]=e;$("ck").classList.remove("hidden");$("ck-steps-card").classList.remove("hidden");if(!$("ck-start-img").getAttribute("src"))$("ck-start-card").classList.add("hidden");ckDraw(e.seq)}
  else if(e.kind=="decision"&&ckChecks[e.check]){ckDecisions[e.check]=e;ckDraw(e.check)}}}catch(e){}}
let timer;async function poll(){clearTimeout(timer);ckPoll();try{const s=await (await fetch("/state?since="+next)).json();
 s.lines.forEach(addLine);next=s.next;$("cmd").textContent=s.cmd?"$ "+s.cmd:"";$("logpath").textContent=s.log;
 const p=$("pill");if(s.running){p.textContent="running";p.className="pill run"}else if(s.code===null){p.textContent="idle";p.className="pill"}
 else{p.textContent="exit "+s.code;p.className="pill "+(s.code==0?"ok":"bad")}
 $("elapsed").textContent=s.cmd?s.elapsed+" s":"";$("run").disabled=s.running;$("home").disabled=s.running;$("stop").disabled=!s.running;wasRunning=s.running}catch(e){}
 timer=setTimeout(poll,wasRunning?400:1500)}
init();
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8890)
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"pizza UI on http://{args.host}:{args.port}  (logs in {LOG_DIR.relative_to(ROOT)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        RUNNER.stop()


if __name__ == "__main__":
    main()
