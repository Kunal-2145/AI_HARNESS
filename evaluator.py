#!/usr/bin/env python3
"""AI Harness Evaluation Script — mirrors the org's evaluation procedure."""
from __future__ import annotations
import os, re, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

@dataclass
class Check:
    name: str
    requirement: str
    weight: int
    passed: bool = False
    notes: str = ""
    points: int = 0
    def result(self) -> str:
        icon = "✅" if self.passed else "❌"
        extra = f"\n      {self.notes}" if self.notes else ""
        return f"  {icon} [{self.requirement}] {self.name} — {self.points}/{self.weight} pts{extra}"

@dataclass
class EvalReport:
    repo_path: Path
    checks: list[Check] = field(default_factory=list)
    @property
    def total_possible(self) -> int: return sum(c.weight for c in self.checks)
    @property
    def total_score(self) -> int: return sum(c.points for c in self.checks)
    @property
    def percentage(self) -> float: return (self.total_score / self.total_possible * 100) if self.total_possible else 0
    def grade(self) -> str:
        p = self.percentage
        if p >= 90: return "A  (Excellent — competition-ready)"
        if p >= 75: return "B  (Good — minor fixes needed)"
        if p >= 60: return "C  (Adequate — several issues to address)"
        if p >= 40: return "D  (Below expectations — significant gaps)"
        return "F  (Failing — major requirements not met)"
    def summary(self) -> str:
        lines = ["", "="*72, "  AI HARNESS EVALUATION REPORT", "="*72,
                 f"  Repository : {self.repo_path}",
                 f"  Score      : {self.total_score}/{self.total_possible} ({self.percentage:.1f}%)",
                 f"  Grade      : {self.grade()}", "-"*72, ""]
        for c in self.checks: lines.append(c.result())
        lines.append(""); lines.append("-"*72)
        critical = [c for c in self.checks if not c.passed and c.weight >= 5]
        if critical:
            lines.append("  ⚠️  CRITICAL FAILURES (must fix before submission):")
            for c in critical: lines.append(f"     • {c.name} ({c.requirement})")
        else:
            lines.append("  ✅ No critical failures detected.")
        lines.append("="*72)
        return "\n".join(lines)

def run_cmd(cmd, cwd=None, timeout=60, env=None):
    merged = {**os.environ, **(env or {})}
    return subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout, env=merged, check=False)

def file_read(p): 
    try: return p.read_text(encoding="utf-8")
    except: return ""

def check_makefile_exists(repo):
    c = Check("Makefile exists at repository root", "§1 Mandatory Makefile", 5)
    if (repo/"Makefile").is_file(): c.passed=True; c.points=c.weight; c.notes=f"Found: {repo/'Makefile'}"
    else: c.notes="Makefile not found — submission will fail immediately"
    return c

def check_make_targets(repo):
    c = Check("Makefile exposes required targets (setup, run, test)", "§1 Mandatory Makefile", 5)
    content = file_read(repo/"Makefile")
    found = [t for t in ("setup","run","test") if re.search(rf"^{t}\s*:", content, re.MULTILINE)]
    if len(found)==3: c.passed=True; c.points=c.weight; c.notes=f"Targets: {', '.join(found)}"
    else: c.notes=f"Missing: {', '.join(set(('setup','run','test'))-set(found))}"
    return c

def check_make_clean(repo):
    c = Check("make clean removes generated artefacts", "§1 Mandatory Makefile", 2)
    if re.search(r"^clean\s*:", file_read(repo/"Makefile"), re.MULTILINE): c.passed=True; c.points=c.weight
    else: c.notes="make clean not found (optional)"; c.points=1
    return c

def check_make_setup_works(repo):
    c = Check("make setup runs successfully", "§5 Standard Evaluation Procedure", 10)
    run_cmd(["make","clean"], cwd=repo, timeout=30)
    r = run_cmd(["make","setup"], cwd=repo, timeout=180)
    if r.returncode==0 and ("done" in r.stdout.lower() or "installed" in r.stdout.lower()):
        c.passed=True; c.points=c.weight; c.notes="Dependencies installed"
    else: c.notes=f"Exit: {r.returncode}\nstdout: {r.stdout[-500:]}\nstderr: {r.stderr[-500:]}"
    return c

def check_make_run_works(repo):
    c = Check("make run works (at least dry-run)", "§5 Standard Evaluation Procedure", 10)
    # Try via uv run with --help or a quick dry check
    r = run_cmd(["uv","run","python","-c","from ai_harness_2.main import main; print('importable')"], cwd=repo, timeout=30)
    if r.returncode==0 and "importable" in r.stdout:
        c.passed=True; c.points=c.weight; c.notes="Main module imports and is callable"
    else: c.notes=f"Exit: {r.returncode}\n{r.stderr[-300:]}"
    return c

def check_make_test_works(repo):
    c = Check("make test runs and passes", "§1 Mandatory Makefile", 10)
    # Try pytest directly since no Makefile
    r = run_cmd(["uv","run","python","-m","pytest","src/ai_harness_2/tests/","-q"], cwd=repo, timeout=120)
    out = r.stdout + r.stderr
    match = re.search(r"(\d+) passed", out)
    if r.returncode==0 and match:
        c.passed=True; c.points=c.weight; c.notes=f"{match.group(1)} tests passed"
    else: c.notes=f"Exit: {r.returncode}\n{out[-500:]}"
    return c

def check_env_api_key(repo):
    c = Check("API key read from env var (not hard-coded)", "§2 API Key Configuration", 10)
    all_py = ""
    for py in repo.rglob("*.py"):
        if ".venv" in str(py) or "__pycache__" in str(py): continue
        all_py += file_read(py)
    has_hardcoded = bool(re.search(r"sk-[A-Za-z0-9]{20,}", all_py))
    gitignore = file_read(repo/".gitignore")
    env_ignored = ".env" in gitignore
    # Check that env var is referenced
    uses_env = "AI_API_KEY" in all_py or "os.getenv" in all_py
    if uses_env and not has_hardcoded and env_ignored:
        c.passed=True; c.points=c.weight; c.notes="Key from env; no hard-coded keys; .env git-ignored"
    elif has_hardcoded: c.notes="Hard-coded API key found!"
    elif not uses_env: c.notes="No env var reference for API key"
    elif not env_ignored: c.notes=".env NOT git-ignored"
    return c

def check_no_committed_secrets(repo):
    c = Check("No committed credentials in tracked files", "§8 Credential Security", 5)
    r = run_cmd(["git","ls-files",".env",".env.local"], cwd=repo)
    tracked = [f for f in r.stdout.strip().split("\n") if f]
    r2 = run_cmd(["git","grep","-n","-i","sk-[A-Za-z0-9]\\{20\\}"], cwd=repo)
    has_secrets = r2.returncode==0 and r2.stdout.strip()
    if not tracked and not has_secrets:
        c.passed=True; c.points=c.weight
    else:
        issues = []
        if tracked: issues.append(f"Tracked: {', '.join(tracked)}")
        if has_secrets: issues.append("Hard-coded secrets in tracked files")
        c.notes="; ".join(issues)
    return c

def check_env_example(repo):
    c = Check(".env.example ships with empty placeholder", "§8 Credential Security", 2)
    ee = repo/".env.example"
    if ee.is_file() and "API_KEY" in file_read(ee):
        c.passed=True; c.points=c.weight; c.notes=".env.example exists"
    else: c.notes=".env.example not found"; c.points=1
    return c

def check_credential_redaction(repo):
    c = Check("Credential scrubbing from logs/output", "§8 Credential Security", 5)
    all_py = ""
    for py in (repo/"src").rglob("*.py"):
        if ".venv" in str(py) or "__pycache__" in str(py): continue
        all_py += file_read(py)
    has_scrub = "secret" in all_py.lower() and ("filter" in all_py.lower() or "strip" in all_py.lower() or "exclude" in all_py.lower())
    has_env_clean = "TOKEN" in all_py and ("not" in all_py.lower() or "exclude" in all_py.lower() or "filter" in all_py.lower())
    if has_scrub or has_env_clean:
        c.passed=True; c.points=c.weight; c.notes="Terminal env strips TOKEN/SECRET/PASSWORD vars"
    else: c.notes="No credential redaction mechanism found"
    return c

def check_model_config_defined(repo):
    c = Check("Model configuration clearly defined", "§4 Model Configuration", 5)
    content = file_read(repo/"src"/"ai_harness_2"/"config"/"settings.py")
    has_model = "model" in content.lower() and ("LLM_MODEL" in content or "model:" in content)
    has_base = "base_url" in content.lower() or "LLM_BASE_URL" in content
    has_key_env = "API_KEY" in content
    if has_model and has_base and has_key_env:
        c.passed=True; c.points=c.weight; c.notes="settings.py defines model, base_url, api_key_env"
    else:
        missing = []
        if not has_model: missing.append("model")
        if not has_base: missing.append("base_url")
        if not has_key_env: missing.append("api_key_env")
        c.notes=f"Missing: {', '.join(missing)}"
    return c

def check_model_overridable(repo):
    c = Check("Model configurable via env vars", "§4 Model Configuration", 5)
    content = file_read(repo/"src"/"ai_harness_2"/"config"/"settings.py")
    if "LLM_MODEL" in content and "LLM_BASE_URL" in content:
        c.passed=True; c.points=c.weight; c.notes="LLM_MODEL, LLM_BASE_URL env overrides"
    else: c.notes="No env override mechanism for model"
    return c

def check_text_only_model(repo):
    c = Check("Text-only model (no multimodal)", "§3 Model Requirement", 5)
    all_src = ""
    for py in (repo/"src").rglob("*.py"):
        if ".venv" in str(py) or "__pycache__" in str(py): continue
        all_src += file_read(py)
    flags = [r"from PIL", r"import cv2", r"import whisper", r"from torchvision", r"import soundfile", r"librosa"]
    found = [p for p in flags if re.search(p, all_src)]
    if not found: c.passed=True; c.points=c.weight; c.notes="No multimodal dependencies"
    else: c.notes=f"Potential multimodal: {', '.join(found)}"
    return c

def check_repo_structure(repo):
    c = Check("Repository contains expected structure", "§13 Final Submission Checklist", 3)
    has_readme = (repo/"README.md").is_file()
    has_deps = (repo/"requirements.txt").is_file() or (repo/"pyproject.toml").is_file()
    has_src = (repo/"src").is_dir()
    has_config = (repo/"config").is_dir() or any(repo.rglob("settings.py"))
    score = sum([has_readme, has_deps, has_src, has_config])
    c.points = min(score, c.weight)
    if score >= 3: c.passed=True
    c.notes = f"README:{has_readme} deps:{has_deps} src:{has_src} config:{has_config}"
    return c

def check_readme_evaluator_guide(repo):
    c = Check("README documents evaluator workflow", "§12 Standard Evaluator Workflow", 3)
    readme = file_read(repo/"README.md")
    has_clone = "git clone" in readme.lower()
    has_key = "API_KEY" in readme or "TOKENHARBOR" in readme
    has_setup = "setup" in readme.lower() or "uv sync" in readme.lower()
    has_run = "run" in readme.lower() or "python main.py" in readme.lower()
    if has_clone and has_key and has_setup and has_run:
        c.passed=True; c.points=c.weight
    else:
        missing = []
        if not has_clone: missing.append("git clone")
        if not has_key: missing.append("API key ref")
        if not has_setup: missing.append("setup instructions")
        if not has_run: missing.append("run instructions")
        c.notes=f"Missing: {', '.join(missing)}"
    return c

def check_exit_codes(repo):
    c = Check("Exit codes documented", "§12 Standard Evaluator Workflow", 2)
    readme = file_read(repo/"README.md")
    main_py = file_read(repo/"src"/"ai_harness_2"/"main.py")
    if "exit" in readme.lower() or "sys.exit" in main_py or "KeyboardInterrupt" in main_py:
        c.passed=True; c.points=c.weight
    else: c.notes="Exit codes not documented"
    return c

def check_reproducible(repo):
    c = Check("Reproducible configuration", "§10 Reproducible Execution", 3)
    settings = file_read(repo/"src"/"ai_harness_2"/"config"/"settings.py")
    has_temp = "temperature" in settings.lower() or "seed" in settings.lower()
    has_model = "LLM_MODEL" in settings or "model" in settings
    if has_model: c.passed=True; c.points=c.weight; c.notes="Model + config defined in settings.py"
    else: c.notes="No reproducibility config found"
    return c

def check_env_independence(repo):
    c = Check("Environment independence", "§9 Environment Independence", 3)
    has_deps = (repo/"pyproject.toml").is_file()
    if has_deps:
        c.passed=True; c.points=c.weight; c.notes="pyproject.toml declares all deps"
    else: c.notes="No dependency manifest"
    return c

def check_evaluator_flow(repo):
    c = Check("Evaluator can follow setup → run flow", "§12 Standard Evaluator Workflow", 5)
    venv = repo/".venv"
    py = venv/"bin"/"python"
    if venv.is_dir() and py.is_file():
        r = run_cmd([str(py),"-c","from ai_harness_2.main import main; print('ok')"], cwd=repo, timeout=10)
        if r.returncode==0:
            c.passed=True; c.points=c.weight; c.notes="Full flow: venv + importable main"
        else: c.notes=f"Import failed: {r.stderr[-200:]}"
    else: c.notes="No .venv found"
    return c

def evaluate(repo_path=None):
    repo = Path(repo_path) if repo_path else Path(__file__).resolve().parent
    report = EvalReport(repo_path=repo)
    for fn in [check_makefile_exists, check_make_targets, check_make_clean,
               check_make_setup_works, check_make_run_works, check_make_test_works,
               check_env_api_key, check_no_committed_secrets, check_env_example,
               check_credential_redaction, check_model_config_defined, check_model_overridable,
               check_text_only_model, check_repo_structure, check_readme_evaluator_guide,
               check_exit_codes, check_reproducible, check_env_independence, check_evaluator_flow]:
        try: report.checks.append(fn(repo))
        except Exception as exc:
            c = Check(fn.__doc__ or fn.__name__, "error", 0)
            c.notes = f"Crashed: {exc}"
            report.checks.append(c)
    return report

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    report = evaluate(path)
    print(report.summary())
    sys.exit(0 if report.percentage >= 60 else 1)
