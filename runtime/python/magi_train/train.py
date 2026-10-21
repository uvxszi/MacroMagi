"""Running a training plan.

The plan (JSON, written by the Magic tab) says what kind of model to make,
where to start, its architecture, the datasets and the stages, each with the
optimizer wired into it. Progress goes out as JSON lines; the result goes to
`out_dir`: `weights.safetensors`, `model.onnx` and `report.json`.

Stages:

    pretrain     next-token prediction on plain text, optionally growing the
                 context from short to full (a curriculum)
    sft          fine-tuning on requests and answers (fully, or with LoRA or DoRA
                 adapters that are folded back in afterwards); answers *not* to
                 give are made less likely (unlikelihood)
    rl           reinforcement learning: several answers per request are scored
                 by the plan's reward rules, and the better-than-average ones
                 are made more likely (group-relative policy gradient)
    compaction   learning to turn a long context into a compact one
    vision       training a picture classifier; negative pictures of a class
                 are made less likely to be called that class

A stage with a `checkpoint` ({"path", "meta"}) saves the weights after it, so a
later training can start from there. On a graphics card the model trains in
bfloat16 with full-precision weights (MAGI_TRAIN_PRECISION=fp32 turns that
off); fine-tuning packs several requests into each row; datasets are mixed by
the square root of their size.
"""

import json
import math
import os
import random
import re
import sys
import threading
import time

# Graphics memory in segments that grow, rather than fixed blocks: far less of it lost between blocks,
# so a step that nearly fills the card does not have to empty PyTorch's cache to fit (Linux only).
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import blocks, corpus, data, optim, safetensors  # noqa: E402
from .tokens import Tokenizer  # noqa: E402


# The run's log, beside its saves (`out_dir/train.log`): what the console showed, kept when it is gone.
LOG_FILE = None


def emit(event, **fields):
    fields["event"] = event
    line = json.dumps(fields) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass


def log(message):
    emit("log", message=message)


# When this run began, for the times the console gives after each stage.
STARTED = time.time()
STAGE_NAMES = {"pretrain": "Pretraining", "sft": "Fine-tuning", "rl": "Reinforcement learning",
               "compaction": "Context compaction", "vision": "Vision training"}


def duration(seconds):
    """A span of time, short: 45 s, 3 min 05 s, 1 h 12 min."""
    seconds = int(round(seconds))
    if seconds < 60:
        return "%d s" % seconds
    if seconds < 3600:
        return "%d min %02d s" % divmod(seconds, 60)
    hours, rest = divmod(seconds, 3600)
    return "%d h %02d min" % (hours, rest // 60)


def took(what, since):
    """Logs how long a part took, and the whole run so far."""
    now = time.time()
    log("%s took %s · %s in all." % (what, duration(now - since), duration(now - STARTED)))


def _on_cpu(value):
    """A copy of a state (tensors in dicts and lists) in system memory."""
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _on_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_on_cpu(v) for v in value)
    return value


class Resume:
    """Saves a long run as it goes – the weights, the optimizer, the stage and the step – so it goes on
    where it stopped when it is run again (a Colab session ends, a server restarts). Saved every
    MAGI_TRAIN_SAVE_MINUTES (20) minutes into out_dir/resume.pt, and removed when the run ends. Training
    waits only for a copy in system memory; writing it (several GB, to Google Drive on Colab) goes on
    beside it."""

    def __init__(self, path, fingerprint):
        self.path, self.fingerprint = path, fingerprint
        self.previous = path[:-3] + ".prev.pt" if path.endswith(".pt") else path + ".prev"
        self.every = float(os.environ.get("MAGI_TRAIN_SAVE_MINUTES", "20")) * 60
        self.saved_at = time.time()
        self.stage = 0
        self.state = None
        for candidate in (path, self.previous):
            if not os.path.exists(candidate):
                continue
            state = torch.load(candidate, map_location="cpu", weights_only=False)
            if state.get("fingerprint") != fingerprint:
                log("A saved run in %s is for another plan: it starts over." % candidate)
                break
            if not all(bool(torch.isfinite(v).all()) for v in state["model"].values()
                       if torch.is_tensor(v) and v.is_floating_point()):
                emit("warning", message="The saved run in %s has weights that are not numbers (NaN): %s"
                     % (candidate, "the one before is tried." if candidate == path else "it starts over."))
                continue
            self.state = state
            break
        self.model = None
        self.writing = None

    def done_stages(self):
        return self.state["stage"] if self.state else 0

    def restore(self, optimizer):
        """The optimizer's state and the step to go on from, for the stage that was running."""
        if not self.state or self.state["stage"] != self.stage:
            return 0
        for o, saved in zip(optimizer.optimizers, self.state["optimizer"]):
            o.load_state_dict(saved)
        step = self.state["step"]
        log("Goes on from step %d of this stage (saved %s ago)." % (step, duration(time.time() - self.state["time"])))
        self.state = None
        return step

    def maybe_save(self, step, optimizer):
        if time.time() - self.saved_at < self.every:
            return
        self.saved_at = time.time()
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            emit("warning", message="The weights are not numbers (NaN) at step %d: the run is not saved, so the "
                                    "last save stays good." % step)
            return
        self.wait()
        state = _on_cpu({"fingerprint": self.fingerprint, "stage": self.stage, "step": step, "time": time.time(),
                         "model": self.model.state_dict(), "optimizer": [o.state_dict() for o in optimizer.optimizers]})
        self.writing = threading.Thread(target=self._write, args=(state, step))
        self.writing.start()

    def _write(self, state, step):
        part = self.path + ".part"
        try:
            torch.save(state, part)
            # The save before stays beside it, in case this one turns out bad.
            if os.path.exists(self.path):
                os.replace(self.path, self.previous)
            os.replace(part, self.path)
            log("Saved the run at step %d, to go on from there if it stops." % step)
        except OSError as error:
            emit("warning", message="The run could not be saved (%s)." % error)

    def wait(self):
        """Until the last save is written."""
        if self.writing is not None:
            self.writing.join()
            self.writing = None

    def finish(self):
        self.wait()
        for saved in (self.path, self.previous):
            try:
                os.remove(saved)
            except OSError:
                pass


# Training longer than this on this computer is better done on a GPU server (File › Export as Python).
LOCAL_LIMIT = 20 * 60


class Estimate:
    """How long the run takes, worked out as it goes: every stage's steps, and those done."""

    def __init__(self):
        self.total = 0
        self.done = 0
        self.warned = False

    def at(self, stage, step, steps, first, timed):
        """After a stage's first steps: the time a step takes (over the last `timed`), and what is left."""
        per_step = (time.time() - first) / max(1, timed)
        left = per_step * ((steps - step) + max(0, self.total - self.done - steps))
        whole = time.time() - STARTED + left
        log("%s: %.2f s a step – about %s left, %s in all (if the later stages' steps take as long)."
            % (stage, per_step, duration(left), duration(whole)))
        if whole > LOCAL_LIMIT and not self.warned:
            self.warned = True
            emit("warning", message="This training takes about %s here. On your own computer, a GPU server trains "
                                    "it faster: File › Export as Python, then run the folder on Colab or a rented GPU "
                                    "– the Magic guide says which." % duration(whole))


ESTIMATE = Estimate()


def check_packages(needs_onnx=True):
    """Everything training needs, checked before anything is trained: onnx only when the run
    writes an ONNX file (models MacroMagi's native engine runs need none)."""
    if not needs_onnx:
        return
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit(
            "The onnx package is not installed in this Python (%s). Install it under Settings › Runtime, "
            "then train again." % sys.executable
        )


def pick_devices(choice):
    """The devices to train on: a list of CUDA indices, "auto", or the CPU."""
    if isinstance(choice, list) and choice:
        if torch.cuda.is_available():
            found = [i for i in choice if i < torch.cuda.device_count()]
            if found:
                return [torch.device("cuda", i) for i in found]
        else:
            emit("warning", message="This PyTorch cannot use graphics cards (it is the CPU build), so training "
                                    "runs on the CPU. Install PyTorch for NVIDIA graphics cards under Settings › Runtime.")
        return [torch.device("cpu")]
    if choice == "auto" and torch.cuda.is_available():
        return [torch.device("cuda", 0)]
    return [torch.device("cpu")]


def stay_on_card(devices):
    """Keeps PyTorch within the memory each graphics card has free, less a little for the desktop.
    Past it, NVIDIA's driver on Windows does not fail but lends system memory, about ten times
    slower; failing instead lets training recompute activations or split batches (`run_steps`).
    MAGI_TRAIN_SPILL=1 allows the lending. The GB allowed on the first card, or None."""
    if os.environ.get("MAGI_TRAIN_SPILL") == "1":
        return None
    allowed = None
    for device in devices:
        if device.type != "cuda":
            continue
        free, total = torch.cuda.mem_get_info(device)
        keep = torch.cuda.memory_reserved(device) + free - max(256 << 20, total * 3 // 100)
        keep = max(keep, total // 10)
        torch.cuda.set_per_process_memory_fraction(min(1.0, keep / total), device)
        allowed = keep / 1e9 if allowed is None else allowed
    return allowed


def describe(devices):
    if devices[0].type == "cuda":
        return "Training on %s." % ", ".join(torch.cuda.get_device_name(d) for d in devices)
    return "Training on the CPU (%d threads)." % torch.get_num_threads()


class Runner:
    """A model on one or more devices."""

    def __init__(self, model, devices):
        self.core = model.to(devices[0])
        self.devices = devices
        self.parallel = None
        if len(devices) > 1:
            self.parallel = torch.nn.DataParallel(self.core, device_ids=[d.index for d in devices])

    def __call__(self, *args, **kwargs):
        return (self.parallel or self.core)(*args, **kwargs)

    @property
    def device(self):
        return self.devices[0]


class Precision:
    """How a model trains on a device: bfloat16 on graphics cards that have it (Ampere and
    newer), float16 with a loss scale on older ones (Turing: Colab's T4, Volta), full
    precision on the CPU or with MAGI_TRAIN_PRECISION=fp32. The weights stay in full
    precision either way."""

    def __init__(self, device):
        self.dtype = None
        if device.type == "cuda" and os.environ.get("MAGI_TRAIN_PRECISION", "") != "fp32":
            major, _ = torch.cuda.get_device_capability(device)
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.scale = 2.0 ** 14 if self.dtype == torch.float16 else 1.0
        self.good = 0

    def describe(self):
        if self.dtype == torch.bfloat16:
            return ("Training in mixed precision: the math in bfloat16 (the graphics card's fastest), the weights "
                    "kept in float32 so small updates are not lost.")
        if self.dtype == torch.float16:
            return ("Training in mixed precision: the math in float16 with a loss scale (this graphics card has no "
                    "bfloat16), the weights kept in float32 so small updates are not lost.")
        return None

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype or torch.float32, enabled=self.dtype is not None)

    def backward(self, loss):
        (loss * self.scale).backward()

    def unscale(self, params):
        """Takes the loss scale back off the gradients; False when they overflowed (the step is skipped)."""
        if self.scale == 1.0:
            return True
        grads = [p.grad for p in params if p.grad is not None]
        for g in grads:
            g.div_(self.scale)
        finite = bool(torch.stack([g.abs().max() for g in grads]).isfinite().all()) if grads else True
        if not finite:
            self.scale = max(1.0, self.scale / 2)
            self.good = 0
            return False
        self.good += 1
        if self.good % 500 == 0:
            self.scale = min(2.0 ** 24, self.scale * 2)
        return True


FULL = Precision(torch.device("cpu"))


def run_steps(runner, settings, steps, stage_name, compute, counter, after_backward=None, precision=FULL,
              shrink=None, resume=None, uncompile=None):
    """The training loop of a stage: `compute(step)` gives (loss, extra fields) for one batch.
    When the graphics card runs out of memory, `shrink()` makes the work smaller (splitting batches,
    then recomputing activations) and the step runs again; it gives False when it cannot. When memory is
    so nearly full that PyTorch has to empty its cache to fit a step (every step a lot slower), it
    splits batches too (`shrink(True)`: never recomputing). `uncompile(error)` turns compiled layers off
    when compiling them failed (it gives False when they were not the cause), and the step runs again."""
    optimizer = optim.make(runner.core, settings, steps)
    # Going on from a run that stopped: the optimizer as it was, from the step it reached.
    start = resume.restore(optimizer) if resume else 0
    accumulate = max(1, int(settings.get("accumulate", 1)))
    clip = float(settings.get("clip", 1.0))
    offload = bool(settings.get("offload")) and runner.device.type == "cuda"
    runner.core.checkpointing = bool(settings.get("checkpointing")) or getattr(runner.core, "checkpointing", False)
    runner.core.train()
    params = [p for p in runner.core.parameters() if p.requires_grad]
    last = float("nan")
    step = start
    first = None
    # Timed from the sixth step on (the first warm up: kernels chosen, memory found), again after a split.
    timed_from, estimate_at = 5, 25
    cramped = 0
    retries = _alloc_retries(runner.device)
    # Updates skipped in a row because the gradients were not numbers (NaN or infinite), and the last norm.
    broken, norm = 0, float("nan")
    while step < steps:
        try:
            loss, extra = compute(step)
            if loss is not None and loss.requires_grad:
                precision.backward(loss / accumulate)
        except Exception as error:  # noqa: BLE001 - compiling's failures, and memory; anything else is raised
            optimizer.zero_grad()
            loss = None
            if uncompile and uncompile(error):
                torch.cuda.empty_cache()
                emit("warning", message="Compiling the layers failed during training (%s): it goes on without."
                     % str(error).strip().splitlines()[-1][:200])
                continue
            if not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            how = shrink() if shrink else None
            if not how:
                raise
            emit("warning", message="The graphics card ran out of memory: %s, and the step runs again." % how)
            continue
        if after_backward:
            after_backward()
        if (step + 1) % accumulate == 0 or step + 1 == steps:
            if precision.unscale(params):
                norm = float(torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf")))
                if math.isfinite(norm):
                    broken = 0
                    if offload:
                        optim.offload(optimizer, runner.device)
                    optimizer.step(step)
                    if offload:
                        optim.offload(optimizer, torch.device("cpu"))
                else:
                    # One bad step would make every weight NaN, and the next save would keep them so.
                    broken += 1
                    if broken in (1, 10):
                        emit("warning", message="Step %d's gradients were not numbers: its update is skipped."
                             % (step + 1))
                    if broken >= 50:
                        raise SystemExit("Training broke down: 50 updates in a row had gradients that were not "
                                         "numbers. Nothing was saved from them; lower the learning rates and go on "
                                         "from the last save.")
            optimizer.zero_grad()
        if loss is not None:
            last = float(loss.detach())
        counter[0] += 1
        if resume:
            resume.maybe_save(step + 1, optimizer)
        now = _alloc_retries(runner.device)
        if now > retries and step > start and shrink:
            cramped += 1
            if cramped == 2:  # once can be chance; twice, it is every step
                cramped = 0
                how = shrink(True)
                if how:
                    torch.cuda.empty_cache()
                    emit("warning", message="The graphics card's memory is so full that every step waits for it: %s."
                         % how)
                    timed_from, estimate_at, first = step + 5, step + 25, None
            now = _alloc_retries(runner.device)
        retries = now
        if step == timed_from:
            first = time.time()
        elif step == estimate_at and first is not None and steps > 40:
            ESTIMATE.at(STAGE_NAMES.get(stage_name, stage_name), step, steps, first, estimate_at - timed_from)
        if step % 5 == 0 or step + 1 == steps:
            emit("progress", stage=stage_name, step=step + 1, steps=steps,
                 loss=None if math.isnan(last) else round(last, 4),
                 grad_norm=round(norm, 3) if math.isfinite(norm) else None, **(extra or {}))
        step += 1
    return last


def _alloc_retries(device):
    """How many times PyTorch has had to empty its memory cache to find room (graphics cards only)."""
    if device.type != "cuda":
        return 0
    return torch.cuda.memory_stats(device).get("num_alloc_retries", 0)


# ---- compiling (experimental) -------------------------------------------------


class Compiled:
    """torch.compile on each layer of a text model, when it is asked for (the plan's "compile", or
    MAGI_TRAIN_COMPILE=1) and proves itself: one training step is timed and its loss compared, as it is
    and compiled, and compiling is kept only when it gives the same loss and is faster. It needs Linux
    (Triton); anything that fails leaves the model as it was. Only pretraining uses it: its rows all have
    the same length, and every new length compiles again."""

    def __init__(self, model):
        self.model = model
        self.kept = False

    def use(self, on):
        """Compiled layers on (learning) or off (writing, and reinforcement learning)."""
        if self.kept:
            for layer in self.model.layers:
                layer._compiled_call_impl = layer._magi_compiled if on else None

    def trial(self, context, device, precision):
        import platform
        if platform.system() != "Linux" or device.type != "cuda":
            log("Compiling is for Linux with an NVIDIA card (Triton): it trains as it is.")
            return
        try:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
            model = self.model
            g = torch.Generator(device="cpu").manual_seed(0)
            tokens = torch.randint(0, model.embed_tokens.num_embeddings, (1, context + 1), generator=g).to(device)

            def step():
                with precision.autocast():
                    out = model(tokens[:, :-1], targets=tokens[:, 1:], loss=True)
                loss = out["loss"].float().mean() + out["aux"].float().mean()
                precision.backward(loss)
                model.zero_grad(set_to_none=True)
                return float(loss.detach())

            def timed():
                torch.cuda.synchronize()
                began = time.time()
                for _ in range(3):
                    value = step()
                torch.cuda.synchronize()
                return value, (time.time() - began) / 3

            step()
            plain_loss, plain = timed()
            log("Compiling the layers (a few minutes the first time)…")
            for layer in model.layers:
                layer.compile(dynamic=False)
                layer._magi_compiled = layer._compiled_call_impl
            self.kept = True
            step()
            step()
            compiled_loss, fast = timed()
            same = abs(compiled_loss - plain_loss) <= 1e-2 * max(1.0, abs(plain_loss))
            if same and fast < plain * 0.95:
                log("Compiled: %.2f s a row instead of %.2f – kept." % (fast, plain))
                return
            log("Compiled: %.2f s a row instead of %.2f%s – not kept." % (
                fast, plain, "" if same else ", a different loss (%.4f, not %.4f)" % (compiled_loss, plain_loss)))
        except Exception as error:  # noqa: BLE001 - compiling is a trial, never a failure
            emit("warning", message="Compiling failed (%s): it trains as it is." % str(error).splitlines()[0][:200])
        self.use(False)
        self.kept = False


# ---- text -------------------------------------------------------------------


class TextTrainer:
    def __init__(self, plan):
        self.plan = plan
        self.spec = plan["text"]
        self.learn_tokens()
        self.tok = Tokenizer(self.spec)
        architecture = dict(plan["architecture"])
        self.context = int(architecture.get("context", self.spec.get("context", 256)))
        architecture["context"] = self.context
        # Long pieces of work are learned as the agent loop reads them: earlier results shortened to
        # leave room for the longest answer.
        data.BUDGET = max(self.context - int(self.spec.get("max_tokens", 0) or 0), self.context // 2)
        self.architecture = architecture
        self.personas = len(self.spec.get("personas", []))
        self.uses_personas = any((l.get("ffn") or {}).get("kind") == "persona" for l in architecture.get("layers", []))
        model = blocks.TextModel(self.tok.vocab, architecture, self.tok.special(self.spec["assistant_tag"]), self.personas)
        start = plan.get("start", {})
        if start.get("weights"):
            missing = model.load_state_dict(safetensors.load(start["weights"]), strict=False)
            if missing.missing_keys or missing.unexpected_keys:
                raise SystemExit(
                    "The base model's weights do not fit this architecture (%d missing, %d left over). Keep the "
                    "base model's architecture blocks to train it further."
                    % (len(missing.missing_keys), len(missing.unexpected_keys))
                )
            log("Started from the base model's weights.")
        devices = pick_devices(plan.get("devices", "auto"))
        log(describe(devices))
        self.runner = Runner(model, devices)
        # Mixed precision: bfloat16 or float16, the weights kept in full precision.
        self.precision = Precision(devices[0])
        if devices[0].type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.precision.describe():
            log(self.precision.describe())
        allowed = stay_on_card(devices)
        if allowed:
            log("Uses up to %.1f GB of graphics memory (what is free): more would spill into system memory, "
                "about ten times slower, so training saves memory instead." % allowed)
        self.rng = random.Random(plan.get("seed", 0))
        # Rows of a batch run at once; None: all. Lowered when the graphics card runs out of memory.
        self.micro = None
        self.accumulate = 1
        # Whether it learned compaction samples (the AI Agent window then compacts long chats with it).
        self.compacts = False
        log("%s parameters, %d layers, %d tokens of context." % (
            "{:,}".format(blocks.parameter_count(model)), len(model.layers), self.context))
        self.counter = [0]
        self.pending = []
        self.last_loss = float("nan")
        self.stages = []
        fingerprint = json.dumps({"architecture": plan.get("architecture"), "stages": [
            {k: v for k, v in st.items() if k in ("type", "steps", "batch", "node")} for st in plan.get("stages", [])],
            "vocab": self.tok.vocab}, sort_keys=True)
        self.resume = Resume(os.path.join(plan.get("out_dir", "."), "resume.pt"), fingerprint)
        self.resume.model = self.runner.core
        if self.resume.state:
            self.runner.core.load_state_dict(self.resume.state["model"])
            log("A saved run was found: it goes on from stage %d, step %d." % (
                self.resume.state["stage"] + 1, self.resume.state["step"]))
        self.compiled = Compiled(self.runner.core)
        self.stage_compiles = False
        if plan.get("compile") or os.environ.get("MAGI_TRAIN_COMPILE") == "1":
            self.compiled.trial(self.context, self.device, self.precision)
        self.datasets = {key: [] for key in plan.get("datasets", {})}
        # Corpora too big for memory (a Hugging Face dataset), prepared when pretraining needs them.
        self.corpora = {key: [f["corpus"] for f in files if isinstance(f, dict) and "corpus" in f]
                        for key, files in plan.get("datasets", {}).items()}
        # A reasoning effort no sample is written at is learned from the nearest level's samples.
        donors = {}
        if self.spec.get("effort"):
            paths = [path for files in plan.get("datasets", {}).values() for path in files if isinstance(path, str)]
            counts = data.effort_counts(paths, self.spec)
            for level, count in counts:
                donor = data.donor(counts, level)
                if donor:
                    donors[level] = donor
                    log("No samples are at reasoning effort %s: it learns from the %s samples too." % (level, donor))
        for key, files in plan.get("datasets", {}).items():
            for path in files:
                if isinstance(path, str):
                    self.datasets[key].extend(data.read(path, self.tok, donors))

    def learn_tokens(self):
        """Learned tokens not learned yet – a model whose corpus is only read where it trains – are
        learned here, from the dataset files and the first part of each corpus, and written to
        out_dir/tokenizer.json for MacroMagi (and run.py) to keep with the model."""
        tokenizer = self.spec.get("tokenizer") or {}
        if tokenizer.get("kind") != "learned" or tokenizer.get("merges"):
            return
        # Learned before, by this run (it went on after it stopped): the same tokens, not learned again –
        # the corpus and the weights were cut and trained in them.
        saved = os.path.join(self.plan.get("out_dir", "."), "tokenizer.json")
        if os.path.exists(saved):
            with open(saved, encoding="utf-8") as handle:
                learned = json.load(handle)
            if learned.get("size") == tokenizer.get("size") and learned.get("merges"):
                self.spec["tokenizer"] = dict(tokenizer, merges=learned["merges"])
                log("Its tokens are the ones learned before (%s pieces of words)." % "{:,}".format(len(learned["merges"])))
                return
        plain = Tokenizer(dict(self.spec, tokenizer={}))
        texts = []
        for files in self.plan.get("datasets", {}).values():
            for item in files:
                if isinstance(item, str):
                    texts.extend(s.text if s.kind == "text" else s.prompt + " " + s.response
                                 for s in data.read(item, plain, {}))
                elif isinstance(item, dict) and "corpus" in item:
                    log("Reading the start of %s to learn the model's tokens from." % item["corpus"]["repo"])
                    limit = int(os.environ.get("MAGI_TOKENS_SAMPLE_MB", "200")) << 20
                    texts.extend(corpus.sample_texts(item["corpus"], limit))
        merges = Tokenizer.learn(texts, int(tokenizer.get("size", 4096)), self.spec.get("specials", []))
        self.spec["tokenizer"] = dict(tokenizer, merges=merges)
        os.makedirs(self.plan.get("out_dir", "."), exist_ok=True)
        with open(os.path.join(self.plan.get("out_dir", "."), "tokenizer.json"), "w", encoding="utf-8") as handle:
            json.dump(self.spec["tokenizer"], handle)
        log("Learned %s pieces of words for its tokens." % "{:,}".format(len(merges)))

    @property
    def device(self):
        return self.runner.device

    def samples(self, keys, kinds):
        out = []
        for key in keys or []:
            out.extend(s for s in self.datasets.get(key, []) if s.kind in kinds)
        return out

    # -- batches --

    def window(self, ids, length):
        if len(ids) <= length:
            return ids
        start = random.randrange(0, len(ids) - length + 1)
        return ids[start:start + length]

    def ids_of(self, sample):
        """A request's tokens and how many at the end are its answer, encoded once."""
        if sample.ids is None:
            prompt = self.tok.encode(sample.prompt)
            answer = self.tok.encode(self.tok.answer(sample.response))
            if not sample.keep:
                return prompt + answer, len(answer)
            sample.ids = (prompt + answer, len(answer))
        return sample.ids

    def chat_corpus(self, spec):
        """A chat corpus's conversations as requests and answers, as a .jsonl file's are."""
        log("Reading conversations from %s…" % spec["repo"])
        rows = corpus.conversations(spec)
        levels = data.learned_at({}, self.spec) if self.spec.get("effort") else [None]
        out = []
        for messages in rows:
            for level in levels:
                for sample in data._samples_of({"messages": messages}, self.tok, level):
                    sample.source, sample.keep = spec["repo"], False
                    out.append(sample)
        log("%s conversations from %s." % ("{:,}".format(len(rows)), spec["repo"]))
        return out

    def packed(self, mixture, rows):
        """Requests packed one after another into rows, each seeing only itself: inputs, targets
        (only answers are learned) and the samples' numbers. A step has `rows` rows of the
        context's worth of token places, but in rows only as long as its longest request needs (a
        power of two, from 256): packed attention's work grows with a row's length, so short
        requests in 64 rows of 512 instead of 16 of 2048 take a quarter of it. The places stay the
        same, so the memory a step needs does too."""
        places = rows * self.context
        pool, total = [], 0
        while total < places:
            pool.append(self.request(mixture))
            total += len(pool[-1][0])
        size = 256
        while size < max(len(ids) for ids, _ in pool) - 1:
            size *= 2
        size = min(size, self.context)
        length = size + 1
        # Each request into the first row it fits in; more are drawn until a few in a row fit nowhere.
        filled = [[] for _ in range(max(1, places // size))]
        used = [0] * len(filled)
        pool.reverse()
        misses = 0
        while misses < 16:
            ids, learn = pool.pop() if pool else self.request(mixture)
            row = next((r for r, n in enumerate(used) if n + len(ids) <= length), None)
            if row is None:
                misses += 1
                continue
            misses = 0
            filled[row].append((ids, learn))
            used[row] += len(ids)
        xs, ys, segs = [], [], []
        for row in filled:
            ids_row, learn_row, seg_row = [], [], []
            for segment, (ids, learn) in enumerate(row):
                ids_row += ids
                learn_row += [0] * (len(ids) - learn) + [1] * learn
                seg_row += [segment] * len(ids)
            pad = length - len(ids_row)
            xs.append(ids_row[:-1] + [self.tok.end_id] * pad)
            ys.append([t if learn_row[i + 1] and seg_row[i + 1] == seg_row[i] else -100
                       for i, t in enumerate(ids_row[1:])] + [-100] * pad)
            segs.append(seg_row[:-1] + [len(row)] * pad)
        device = self.device
        return (torch.tensor(xs, device=device), torch.tensor(ys, device=device), torch.tensor(segs, device=device))

    def request(self, mixture):
        """A request drawn from the mixture, as ids and how many end ones are its answer; one longer
        than the context keeps its end, the answer."""
        ids, learn = self.ids_of(mixture.draw(self.rng))
        if len(ids) > self.context + 1:
            cut = len(ids) - (self.context + 1)
            ids, learn = ids[cut:], min(learn, self.context)
        return ids, learn

    def lm_batch(self, stream, batch, length):
        xs, ys = [], []
        for _ in range(batch):
            chunk = stream.window(self.rng, length + 1) if isinstance(stream, corpus.Corpus) else self.window(stream, length + 1)
            pad = length + 1 - len(chunk)
            xs.append(chunk[:-1] + [self.tok.end_id] * pad)
            ys.append(chunk[1:] + [-100] * pad)
        return torch.tensor(xs, device=self.device), torch.tensor(ys, device=self.device)

    def pair_tensors(self, samples):
        """Inputs and targets for requests with answers; only the answer is learned."""
        rows = []
        for sample in samples:
            ids, answer = self.ids_of(sample)
            ids = ids[-(self.context + 1):]
            learn = min(answer, len(ids) - 1)
            rows.append((ids, learn))
        length = max(len(ids) for ids, _ in rows) - 1
        xs, ys = [], []
        for ids, learn in rows:
            pad = length + 1 - len(ids)
            xs.append(ids[:-1] + [self.tok.end_id] * pad)
            ys.append([-100] * (len(ids) - 1 - learn) + ids[len(ids) - learn:] + [-100] * pad)
        return torch.tensor(xs, device=self.device), torch.tensor(ys, device=self.device)

    def outputs(self, x, y=None, segments=None, loss=False):
        labels = None
        if self.uses_personas:
            labels = torch.tensor([self.tok.persona_spans(row) for row in x.tolist()], device=self.device)
        with self.precision.autocast():
            out = self.runner(x, labels, y, segments, loss)
        if out["load"].numel():
            # Applied after the backward pass, so recomputed layers route as they did.
            self.pending.append(out["load"].sum(0).detach())
        return out

    def forward(self, x, y=None, segments=None):
        out = self.outputs(x, y, segments)
        return out["logits"].float(), out["aux"].float().mean()

    def balance(self):
        for load in self.pending:
            self.runner.core.balance(load)
        self.pending = []

    def loss_of(self, x, y, segments=None):
        out = self.outputs(x, y, segments, loss=True)
        return out["loss"].float().mean() + out["aux"].float().mean()

    def learn(self, x, y, segments=None, negative=False, weight=1.0):
        """The loss of a batch – a row at a time in parts when memory is short, each part's
        gradients taken at once (then the loss given back has none left to take)."""
        rows = x.shape[0]
        micro = self.micro or rows
        if micro >= rows:
            loss = self.unlikelihood(x, y) if negative else self.loss_of(x, y, segments)
            return loss * weight
        total = 0.0
        for start in range(0, rows, micro):
            part = slice(start, start + micro)
            seg = segments[part] if segments is not None else None
            loss = self.unlikelihood(x[part], y[part]) if negative else self.loss_of(x[part], y[part], seg)
            share = (min(rows, start + micro) - start) / rows
            self.precision.backward(loss * weight * share / self.accumulate)
            total += float(loss.detach()) * share * weight
        return torch.tensor(total, device=self.device)

    def shrink(self, rows, split_only=False):
        """Less work at once after running out of memory. Each batch of `rows` is split in half first
        – the same work, so no slower while each part still has 8,192 tokens or more to keep the card
        busy – and activations are recomputed (about a third more work) only after that, never when
        `split_only` (memory nearly full, not out)."""
        core = self.runner.core
        micro = self.micro or rows
        if micro > 1 and (core.checkpointing or (micro // 2) * self.context >= 8192):
            self.micro = max(1, micro // 2)
            return "each batch now runs in parts of %d row%s" % (self.micro, "" if self.micro == 1 else "s")
        if split_only:
            return None
        if not core.checkpointing:
            core.checkpointing = True
            return "activations are recomputed from now on"
        return None

    def loop(self, stage, steps, name, compute, rows):
        """A stage's training loop, with this trainer's precision and memory fallbacks."""
        settings = self.settings(stage)
        self.accumulate = max(1, int(settings.get("accumulate", 1)))
        return run_steps(self.runner, settings, steps, name, compute, self.counter, self.balance,
                         precision=self.precision, shrink=lambda split_only=False: self.shrink(rows, split_only),
                         resume=self.resume, uncompile=self.uncompile)

    def uncompile(self, error):
        """Compiled layers off for good when compiling them failed mid-run (often out of memory while
        compiling for a new batch size, which PyTorch reports as the compiler's failure)."""
        if not (self.stage_compiles and self.compiled.kept):
            return False
        if not type(error).__module__.startswith(("torch._dynamo", "torch._inductor")):
            return False
        self.compiled.use(False)
        self.compiled.kept = False
        self.stage_compiles = False
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001 - only frees what compiling kept
            pass
        return True

    def unlikelihood(self, x, y):
        """Makes the tokens of an answer not to give less likely: -log(1 - p)."""
        logits, _ = self.forward(x, None)
        probs = torch.softmax(logits, dim=-1)
        mask = y != -100
        picked = probs.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return -(torch.log((1 - picked).clamp_min(1e-6)) * mask).sum() / mask.sum().clamp_min(1)

    @staticmethod
    def settings(stage):
        return stage.get("optimizer") or {}

    # -- stages --

    def pretrain(self, stage):
        # One stream of tokens per dataset file; files are mixed by the square root of their size.
        streams = {}
        for s in self.samples(stage.get("data"), {"text", "pair"}):
            text = s.text if s.kind == "text" else s.prompt + self.tok.answer(s.response)
            streams.setdefault(s.source, []).extend(self.tok.encode(text) + [self.tok.end_id])
        if not streams and not any(self.corpora.get(key) for key in stage.get("data") or []):
            raise SystemExit("Pretraining has no text: wire a Text Dataset with files into its Text input.")
        streams = list(streams.values())
        # Corpora: written once as token files where the training runs, read through a memory map.
        root = os.environ.get("MAGI_CORPUS_DIR") or os.path.join(self.plan.get("out_dir", "."), "corpus")
        for key in stage.get("data") or []:
            for spec in self.corpora.get(key, []):
                if not corpus.is_chat(spec):
                    streams.append(corpus.prepare(spec, self.tok, root))
        weights = [len(s) ** 0.5 for s in streams]
        steps, batch = int(stage.get("steps", 200)), int(stage.get("batch", 8))
        first = int(stage.get("curriculum_start", 0) or self.context)
        first = max(8, min(first, self.context))
        log("Pretraining on %s tokens of text from %d sources." % ("{:,}".format(sum(map(len, streams))), len(streams)))

        def compute(step):
            length = min(self.context, int(first + (self.context - first) * step / max(1, steps - 1)))
            stream = self.rng.choices(streams, weights=weights, k=1)[0]
            x, y = self.lm_batch(stream, batch, length)
            return self.learn(x, y), ({"context": length} if first < self.context else {})

        self.last_loss = self.loop(stage, steps, "pretrain", compute, batch)

    def sft(self, stage, name="sft", samples=None):
        good = samples if samples is not None else self.samples(stage.get("data"), {"pair"})
        if samples is None:
            # Compaction samples in the data are learned with the rest, mixed in like any file – learned
            # apart, after everything else, a model forgets the rest and writes everything as a compaction.
            compact = [data.compact_pair(self.tok, s) for s in self.samples(stage.get("data"), {"compact"})]
            for sample in compact:
                sample.source = "compaction"
            good = good + compact
            self.compacts = self.compacts or bool(compact)
            # Conversations from a chat corpus (a Hugging Face Dataset of messages), read where it trains.
            for key in stage.get("data") or []:
                for spec in self.corpora.get(key, []):
                    if corpus.is_chat(spec):
                        good = good + self.chat_corpus(spec)
        bad = self.samples(stage.get("negatives"), {"pair", "negative"})
        if not good and not bad:
            raise SystemExit("Fine-tuning has no requests and answers: wire a Text Dataset into its “What to do” input.")
        adapter = stage.get("adapter", "full")
        core = self.runner.core
        if adapter in ("lora", "dora"):
            blocks.add_adapters(core, int(stage.get("rank", 8)), float(stage.get("alpha", 16)), adapter == "dora")
            self.runner = Runner(core, self.runner.devices)
            log("Training %s adapters (rank %s) on %d requests." % (adapter.upper(), stage.get("rank", 8), len(good)))
        else:
            log("Training all weights on %d requests%s." % (len(good), ", %d answers to avoid" % len(bad) if bad else ""))
        steps, batch = int(stage.get("steps", 200)), int(stage.get("batch", 4))
        weight = float(stage.get("negative_weight", 0.5))
        mixture = data.Mixture(good)
        if good:
            files = len(mixture.groups)
            log("Packing requests into %s tokens a step (%d rows of %d), in rows as long as each step's "
                "longest request needs, from %d dataset%s mixed by size."
                % (format(batch * self.context, ","), batch, self.context, files, "" if files == 1 else "s"))

        ahead = []

        def compute(step):
            loss = torch.zeros((), device=self.device)
            if good:
                x, y, segments = ahead.pop() if ahead else self.packed(mixture, batch)
                loss = loss + self.learn(x, y, segments)
                # The next step's rows, packed while the graphics card works on this one.
                ahead.append(self.packed(mixture, batch))
            if bad:
                x, y = self.pair_tensors(self.rng.choices(bad, k=max(1, batch // 2)))
                loss = loss + self.learn(x, y, negative=True, weight=weight)
            return loss, {}

        self.last_loss = self.loop(stage, steps, name, compute, batch)
        if adapter in ("lora", "dora"):
            blocks.merge_adapters(core)
            self.runner = Runner(core, self.runner.devices)

    def compaction(self, stage):
        pairs = [data.compact_pair(self.tok, s) for s in self.samples(stage.get("data"), {"compact"})]
        if not pairs:
            raise SystemExit('Context Compaction needs {"context": …, "compact": …} lines in its dataset.')
        for sample in pairs:
            sample.source = "compaction"
        # The other requests come along, as many again: learned alone, compaction would be all the
        # model remembers – it would end every answer as a compaction.
        others = self.samples(stage.get("data"), {"pair"})
        if others:
            log("Compaction samples, with the dataset's other requests mixed in so the model keeps them.")
        self.compacts = True
        self.sft(stage, "compaction", pairs + others)

    @torch.no_grad()
    def sample(self, prompt_ids, count, max_new, temperature, top_k):
        model = self.runner.core
        model.eval()
        self.compiled.use(False)
        x = torch.tensor([prompt_ids] * count, device=self.device)
        done = torch.zeros(count, dtype=torch.bool, device=self.device)
        # With a cache, each new token is read alone: about as many times faster as the prompt is long.
        cache = {} if model.can_cache() else None
        for _ in range(max_new):
            with self.precision.autocast():
                if cache is None:
                    step = model(x[:, -self.context:])
                else:
                    step = model(x if not cache else x[:, -1:], cache=cache)
                logits = step["logits"][:, -1, :].float() / max(temperature, 1e-3)
            if top_k:
                kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -1e9)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.tok.end_id), nxt)
            x = torch.cat([x, nxt.unsqueeze(-1)], dim=1)
            done = done | (nxt == self.tok.end_id)
            if bool(done.all()):
                break
        model.train()
        self.compiled.use(self.stage_compiles)
        out = []
        for row in x.tolist():
            completion = row[len(prompt_ids):]
            if self.tok.end_id in completion:
                completion = completion[: completion.index(self.tok.end_id) + 1]
            out.append(completion)
        return out

    def tool_name(self, call):
        try:
            parsed = json.loads(call)
        except ValueError:
            return None
        return parsed.get("name") if isinstance(parsed, dict) else None

    def first_tool(self, text):
        tool_open, tool_close = self.spec["tool_tags"]
        found = re.search(re.escape(tool_open) + r"(.*?)" + re.escape(tool_close), text, re.S)
        return self.tool_name(found.group(1)) if found else None

    def offered_tools(self, prompt_text):
        """The tools a prompt offers: its short list (`- name(args): …`) or its JSON one."""
        names = re.findall(r"^- ([A-Za-z_][\w.-]*)\(", prompt_text, re.M)
        names += re.findall(r'"name":\s*"([^"]+)"', prompt_text.split(self.spec["tool_tags"][0])[0])
        return names

    def reward(self, completion, rules, reference, prompt_text=""):
        text = self.tok.decode([t for t in completion if t != self.tok.end_id])
        tool_open, tool_close = self.spec["tool_tags"]
        total = 0.0
        for rule in rules:
            kind, score = rule.get("kind"), float(rule.get("score", 1))
            value = str(rule.get("text", ""))
            if kind == "contains" and value and value in text:
                total += score
            elif kind == "not_contains" and value and value not in text:
                total += score
            elif kind == "regex" and value:
                try:
                    if re.search(value, text):
                        total += score
                except re.error:
                    pass
            elif kind == "exact" and reference is not None and text.strip() == reference.strip():
                total += score
            elif kind in ("tool_valid", "tool_malformed"):
                calls = re.findall(re.escape(tool_open) + r"(.*?)" + re.escape(tool_close), text, re.S)
                good = 0
                for call in calls:
                    try:
                        parsed = json.loads(call)
                        good += isinstance(parsed, dict) and isinstance(parsed.get("name"), str)
                    except ValueError:
                        pass
                broken = text.count(tool_open) != text.count(tool_close)
                if kind == "tool_valid" and calls and good == len(calls) and not broken:
                    total += score
                if kind == "tool_malformed" and (good < len(calls) or broken):
                    total += score
            elif kind == "same_tool":
                # The first tool called (or none), as the dataset's answer.
                if reference is not None and self.first_tool(text) == self.first_tool(reference):
                    total += score
            elif kind == "unknown_tool":
                offered = set(self.offered_tools(prompt_text))
                names = [self.tool_name(c) for c in re.findall(re.escape(tool_open) + r"(.*?)" + re.escape(tool_close), text, re.S)]
                if offered and any(n and n not in offered for n in names):
                    total += score
            elif kind == "made_up_result":
                # A tool turn or result tag in its own answer: results come from the tools.
                tool_turn = self.spec.get("tool_tag") or ""
                if tool_turn and tool_turn in text:
                    total += score
            elif kind == "loop":
                n = int(rule.get("ngram", 4))
                grams = [tuple(completion[i:i + n]) for i in range(max(0, len(completion) - n + 1))]
                if grams and max(grams.count(g) for g in set(grams)) >= int(rule.get("repeats", 3)):
                    total += score
            elif kind == "short" and len(completion) <= int(rule.get("tokens", 32)):
                total += score
            elif kind == "ends" and completion and completion[-1] == self.tok.end_id:
                total += score
        return total

    def rl(self, stage):
        pairs = self.samples(stage.get("data"), {"pair"})
        if not pairs:
            raise SystemExit("Reinforcement Learning needs requests: wire a Text Dataset into its Requests input.")
        rules = stage.get("rules") or [{"kind": "ends", "score": 1}]
        steps, group = int(stage.get("steps", 50)), max(2, int(stage.get("group", 4)))
        max_new = int(stage.get("max_new", 48))
        temperature, top_k = float(stage.get("temperature", 0.9)), int(stage.get("top_k", 40))

        def compute(step):
            sample = random.choice(pairs)
            prompt = self.tok.encode(sample.prompt)[-(self.context - max_new):]
            completions = self.sample(prompt, group, max_new, temperature, top_k)
            rewards = torch.tensor([self.reward(c, rules, sample.response, sample.prompt) for c in completions],
                                   dtype=torch.float)
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            loss = torch.zeros((), device=self.device)
            for completion, adv in zip(completions, advantage.tolist()):
                if not completion or abs(adv) < 1e-6:
                    continue
                ids = (prompt + completion)[-(self.context + 1):]
                x = torch.tensor([ids[:-1]], device=self.device)
                logits, _ = self.forward(x)
                logp = torch.log_softmax(logits[0, -len(completion):], dim=-1)
                taken = logp.gather(-1, torch.tensor(ids[-len(completion):], device=self.device).unsqueeze(-1)).sum()
                loss = loss - adv * taken / len(completion)
            return loss / group, {"reward": round(float(rewards.mean()), 3)}

        self.last_loss = self.loop(stage, steps, "rl", compute, group)

    # -- results --

    def export(self, out_dir):
        """The weights, and an ONNX model when the plan asks for one (MacroMagi's native engine runs
        most text models from their weights alone)."""
        model = self.runner.core.to("cpu")
        model.eval()
        model.checkpointing = False
        safetensors.save(model.state_dict(), os.path.join(out_dir, "weights.safetensors"))
        if self.plan.get("onnx", True):
            tokens = torch.zeros(1, self.context, dtype=torch.long)
            log("Exporting the model as ONNX.")
            torch.onnx.export(
                blocks.Exported(model), (tokens,), os.path.join(out_dir, "model.onnx"),
                input_names=["tokens"], output_names=["logits"], opset_version=17, dynamo=False,
            )
        return blocks.parameter_count(model)

    def checkpoint(self, stage):
        """Saves the weights after a stage, for a later training to start from."""
        target = stage.get("checkpoint") or {}
        path = target.get("path")
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {k: v.detach().to("cpu") for k, v in self.runner.core.state_dict().items()}
        safetensors.save(state, path)
        meta = dict(target.get("meta") or {})
        meta.update({"stage": stage.get("type"), "saved": time.time(), "steps": int(stage.get("steps", 0)),
                     "loss": None if math.isnan(self.last_loss) else round(self.last_loss, 4)})
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        log("Saved the weights after this stage, to start a later training from.")

    def run(self, out_dir):
        ESTIMATE.total = sum(int(stage.get("steps", 0) or 0) for stage in self.plan.get("stages", []))
        for index, stage in enumerate(self.plan.get("stages", [])):
            kind = stage.get("type")
            if index < self.resume.done_stages():
                # Finished before the run stopped: its weights are in the saved run.
                self.stages.append(kind)
                ESTIMATE.done += int(stage.get("steps", 0) or 0)
                continue
            self.resume.stage = index
            began = time.time()
            # Compiled layers (when kept) for pretraining only, whose rows all have the same length:
            # fine-tuning packs rows as long as each step needs, and every new length compiles again.
            self.stage_compiles = kind == "pretrain"
            self.compiled.use(self.stage_compiles)
            emit("stage", stage=kind)
            {"pretrain": self.pretrain, "sft": self.sft, "rl": self.rl, "compaction": self.compaction}[kind](stage)
            self.stages.append(kind)
            ESTIMATE.done += int(stage.get("steps", 0) or 0)
            self.checkpoint(stage)
            took(stage.get("name") or STAGE_NAMES.get(kind, kind), began)
        began = time.time()
        self.compiled.use(False)
        parameters = self.export(out_dir)
        self.resume.finish()
        took("Saving the model", began)
        return {"stages": self.stages, "steps": self.counter[0], "loss": self.last_loss, "parameters": parameters,
                "compacts": self.compacts}


# ---- vision -----------------------------------------------------------------


def read_pictures(path):
    """Pictures the Magic tab prepared: a JSON header line, then float32 tensors [3, h, w] each."""
    with open(path, "rb") as handle:
        header = json.loads(handle.readline())
        body = bytearray(handle.read())
    w, h = header["width"], header["height"]
    tensor = torch.frombuffer(body, dtype=torch.float32).reshape(len(header["items"]), 3, h, w).clone()
    return header, tensor


class VisionWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return {"logits": self.model(x)}


def train_vision(plan, out_dir):
    devices = pick_devices(plan.get("devices", "auto"))
    log(describe(devices))
    header, pictures = read_pictures(plan["vision"]["pictures"])
    classes = int(plan["vision"]["classes"])
    model = blocks.VisionModel(classes, plan["architecture"])
    if plan.get("start", {}).get("weights"):
        model.load_state_dict(safetensors.load(plan["start"]["weights"]))
        log("Started from the base model's weights.")
    runner = Runner(VisionWrap(model), devices)
    labels = torch.tensor([item["class"] for item in header["items"]])
    negative = torch.tensor([bool(item.get("negative")) for item in header["items"]])
    stage = next((s for s in plan.get("stages", []) if s.get("type") == "vision"), {})
    steps, batch = int(stage.get("steps", 300)), int(stage.get("batch", 16))
    emit("stage", stage="vision")
    log("%d pictures of %d answers." % (len(pictures), classes))
    began = time.time()
    counter = [0]

    def compute(step):
        index = torch.randint(0, len(pictures), (min(batch, len(pictures)),))
        x = pictures[index].to(runner.device)
        # A little shifting, so the model does not learn exact pixel positions.
        x = torch.roll(x, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))
        y, neg = labels[index].to(runner.device), negative[index].to(runner.device)
        logits = runner(x)["logits"]
        loss = torch.zeros((), device=runner.device)
        if (~neg).any():
            loss = loss + F.cross_entropy(logits[~neg], y[~neg])
        if neg.any():
            p = torch.softmax(logits[neg], -1).gather(-1, y[neg].unsqueeze(-1)).squeeze(-1)
            loss = loss - torch.log((1 - p).clamp_min(1e-6)).mean()
        return loss, {}

    loss_value = run_steps(runner, stage.get("optimizer") or {}, steps, "vision", compute, counter)
    model.eval().to("cpu")
    safetensors.save(model.state_dict(), os.path.join(out_dir, "weights.safetensors"))
    sample = torch.zeros(1, 3, header["height"], header["width"])
    took("Vision training", began)
    log("Exporting the model.")
    torch.onnx.export(model, (sample,), os.path.join(out_dir, "model.onnx"), input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    return {"stages": ["vision"], "steps": counter[0], "loss": loss_value, "parameters": blocks.parameter_count(model)}


def main(path):
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    out_dir = plan["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    global LOG_FILE
    LOG_FILE = os.path.join(out_dir, "train.log")
    random.seed(plan.get("seed", 0))
    torch.manual_seed(plan.get("seed", 0))
    global STARTED
    STARTED = began = time.time()
    try:
        check_packages(plan.get("kind") == "vision" or plan.get("onnx", True))
        if plan.get("kind") == "vision":
            report = train_vision(plan, out_dir)
        else:
            report = TextTrainer(plan).run(out_dir)
    except SystemExit as stop:
        emit("error", message=str(stop))
        return 1
    except torch.cuda.OutOfMemoryError:
        emit("error", message="The graphics card ran out of memory. Make the model or the batch smaller, turn on "
                              "“Recompute activations” in the Optimizer block, or train on the CPU.")
        return 1
    report["seconds"] = round(time.time() - began, 1)
    if isinstance(report.get("loss"), float) and math.isnan(report["loss"]):
        report["loss"] = None
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    emit("done", **report)
    return 0
