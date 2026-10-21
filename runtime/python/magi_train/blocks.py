"""The building blocks of models trained in MacroMagi.

A text model is an embedding, a stack of layers and a head that gives a score
per token. The Magic tab's canvas describes it as JSON (`architecture`):

    width, heads, kv_heads    numbers per token, attention heads, key/value heads
    context                   the longest text, in tokens
    positions                 "rope" (rotary, the default) or "learned"
    rope_theta                the rotary base
    hyper_connections         {"streams": n, "sinkhorn": iterations} or absent
    mtp                       {"weight": 0.3} or absent
    prefix_split              true: the request is read both ways at once
    tied                      true: the output head shares the embedding table
    embed_scale               true: embeddings are scaled by sqrt(width) (Gemma)
    layers                    one entry per layer: {"attention": {...}, "ffn": {...},
                              "norm": "pre" | "sandwich"}; "sandwich" also normalizes
                              each part's output, before it is added back (Gemma, GLM)

Attention ("attention.kind"):

    full      causal attention with rotary positions and grouped key/value heads
              (the Llama layout).
              Options: "qk_norm" (RMSNorm on each head's queries and keys, as in
              Qwen3 and Gemma 3), "bias" (on queries, keys and values), "rotary"
              (the share of each head that turns, 1.0 by default; GLM uses 0.5)
              and "window" (look only at the last `window` tokens; 0: all)
    window    the same, looking only at the last `window` tokens
    mla       multi-head latent attention (DeepSeek-V2/V3): keys and values come
              up from a small latent, with a separate rotary part shared by heads
    csa       compressed sparse attention (DeepSeek-V4): every `compress` tokens
              become one entry (a learned, softmax-weighted pooling over two
              overlapping blocks); a lightning indexer picks the `top_k` entries
              each token attends to, beside a sliding window of plain tokens
    hca       heavily compressed attention (DeepSeek-V4): larger blocks, attended
              to densely, beside a sliding window
    ssm       a Mamba-2 state-space layer, in its exact quadratic form

Feed-forward ("ffn.kind"): "dense" (SwiGLU, or GeGLU with "activation": "geglu"), "moe" (DeepSeekMoE: shared and
routed experts; softmax, sigmoid or sqrt-softplus affinities; an auxiliary
balance loss or auxiliary-loss-free bias balancing; or hash routing by token)
and "persona" (an expert per persona of the reasoning, routed by its tags).

A vision model is a stack of convolutions and a classifier.

The forward pass returns a dictionary – logits (or, asked for the loss, the
loss alone), an auxiliary loss and the experts' loads – so that several
graphics cards (DataParallel) can share it;
`Exported` wraps a model as tokens -> logits for ONNX. Top-k choices are
made by rank in exported models, so ONNX runtimes need no TopK operator.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

NEG = -1e4


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # In full precision, also when the rest runs in bfloat16; PyTorch's fused kernel when training.
        h = x.float()
        if self.training and hasattr(F, "rms_norm"):
            out = F.rms_norm(h, (h.shape[-1],), self.weight, self.eps)
        else:
            out = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        # Under autocast, handed on in its half precision: what reads it would cast it anyway, and a
        # float32 copy kept for the backward pass costs twice the memory.
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            return out.to(torch.get_autocast_dtype("cuda"))
        return out.type_as(x)


# Positions the output head's loss reads at once: 8192 x a 4K vocabulary is about 130 MB of scores.
SLICE = 8192


class SlicedLoss(torch.autograd.Function):
    """Cross-entropy of the output head over hidden states [n, width], a slice of positions at a
    time, with the gradients worked out in the same pass (as Liger's fused linear cross-entropy):
    the scores of every position at once – several GB at 16 rows of 2048 tokens – are never kept,
    and nothing is recomputed. Gradients are kept unscaled and scaled in the backward pass, so a
    float16 loss scale reaches them."""

    @staticmethod
    def forward(ctx, h, weight, targets, dtype):
        count = targets.numel()
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        with torch.autocast(h.device.type, enabled=False):
            w = weight.to(dtype)
            for start in range(0, count, SLICE):
                hs, ts = h[start : start + SLICE].to(dtype), targets[start : start + SLICE]
                rows = torch.arange(ts.numel(), device=ts.device)
                # One fused softmax, in float32, read straight from the half-precision scores (a third
                # faster than logsumexp and exp over a float32 copy). A probability below 1e-30 counts
                # as 1e-30 in the loss shown (69 at most); its gradient is exact.
                probs = torch.softmax(hs @ w.t(), dim=-1, dtype=torch.float32)
                picked = probs[rows, ts]
                total -= torch.log(picked.clamp_min(1e-30)).sum()
                # The gradient of the summed loss by the scores: the probabilities, less one at the target.
                probs[rows, ts] = picked - 1.0
                d = probs.to(dtype)
                grad_h[start : start + SLICE] = (d @ w).to(h.dtype)
                if dtype == torch.float16:
                    grad_w.addmm_(d.t().float(), hs.float())  # float16 would overflow summing 8192 rows
                else:
                    grad_w += (d.t() @ hs).float()
        ctx.save_for_backward(grad_h, grad_w)
        ctx.count = count
        ctx.weight_dtype = weight.dtype
        return total / count

    @staticmethod
    def backward(ctx, grad):
        grad_h, grad_w = ctx.saved_tensors
        scale = grad.float() / ctx.count
        return (grad_h.float() * scale).to(grad_h.dtype), (grad_w * scale).to(ctx.weight_dtype), None, None


def _turn(x, cos, sin):
    half = x.shape[-1] // 2
    turned = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + turned * sin


class _Rotate(torch.autograd.Function):
    """Rotary positions keeping nothing but the tables for the backward pass: the gradient of a
    rotation is the rotation the other way (sin negated)."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _turn(x, cos.to(x.dtype), sin.to(x.dtype))

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        return _turn(grad, cos.to(grad.dtype), (-sin).to(grad.dtype)), None, None


def rotate(x, cos, sin):
    """Rotary positions on the last dimension of x [..., n, dim] (halves layout)."""
    if torch.is_grad_enabled() and x.requires_grad and not torch.onnx.is_in_onnx_export():
        return _Rotate.apply(x, cos, sin)
    return _turn(x, cos, sin)


def rank_mask(scores, k):
    """True for the k highest scores on the last axis (ties: the earlier wins), without TopK."""
    n = scores.shape[-1]
    idx = torch.arange(n, device=scores.device)
    higher = (scores.unsqueeze(-1) < scores.unsqueeze(-2)).sum(-1)
    tied = ((scores.unsqueeze(-1) == scores.unsqueeze(-2)) & (idx.view(n, 1) > idx.view(1, n))).sum(-1)
    return (higher + tied) < k


class Context:
    """What every layer needs besides its input: tokens, the mask, rotary tables."""

    def __init__(self, tokens, mask, theta, labels=None, rotary_on=True, causal=True, cache=None):
        self.tokens, self.mask, self.theta, self.labels = tokens, mask, theta, labels
        self.rotary_on = rotary_on
        # Writing with a cache (TextModel.forward's `cache`): the keys and values of the tokens before,
        # and how many there are – the new tokens' first position.
        self.cache = cache
        self.offset = cache.get("length", 0) if cache is not None else 0
        # Plain causal attention: no request read both ways (Encoder/Decoder Split), and
        # no packed samples whose tokens must not see each other.
        self.causal = causal
        self._tables = {}

    def rope(self, dim, positions):
        """cos and sin for `positions` (a 1-D tensor), each [len, dim]."""
        if dim not in self._tables:
            # Compressed entries sit at the end of their (padded) block: a little past the text.
            span = self.offset + self.mask.shape[-1] + 1024
            inv = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim))
            angles = torch.outer(torch.arange(span, dtype=torch.float32, device=positions.device), inv)
            angles = torch.cat([angles, angles], dim=-1)
            if not self.rotary_on:
                angles = torch.zeros_like(angles)
            self._tables[dim] = (angles.cos(), angles.sin())
        cos, sin = self._tables[dim]
        return cos[positions], sin[positions]


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # cuDNN's fused attention first (on an RTX 5070 about twice as fast as the memory-efficient kernel
    # PyTorch picks by itself, with the same results to bfloat16's precision), then FlashAttention;
    # PyTorch moves on to the next one when a kernel cannot take the inputs.
    _PREFERRED = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                  SDPBackend.MATH]
    sdpa_kernel(_PREFERRED, set_priority=True)
except (ImportError, TypeError, AttributeError):  # older PyTorch: its own choice
    _PREFERRED = None


def attend(q, k, v, **options):
    """PyTorch's fused attention, with the fastest kernel this graphics card has. cuDNN makes a plan for
    each new length, so it is asked only for lengths that come again – multiples of 256: pretraining
    rows, packed fine-tuning rows, window chunks; writing a token at a time and reinforcement learning's
    answers (every length new) keep PyTorch's own choice."""
    if _PREFERRED is None or not q.is_cuda or torch.onnx.is_in_onnx_export():
        return F.scaled_dot_product_attention(q, k, v, **options)
    n = q.shape[-2]
    if (n % 256 and n >= 1024 and k.shape[-2] == n and options.get("is_causal")
            and options.get("attn_mask") is None):
        # Causal over a length that is not a multiple of 256 (multi-token prediction's layer, a long
        # answer): padded at the end, which no earlier token sees – about twice as fast.
        extra = -n % 256
        q, k, v = (F.pad(t, (0, 0, 0, extra)) for t in (q, k, v))
        return attend(q, k, v, **options)[..., :n, :]
    if q.shape[-2] % 256 or k.shape[-2] % 256:
        return F.scaled_dot_product_attention(q, k, v, **options)
    with sdpa_kernel(_PREFERRED, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, **options)


def window_mask(mask, window):
    """A local window: each token sees its own block of `window` tokens and the whole block before
    (between `window` and twice as many tokens back). Block-aligned, so training runs it with the
    fast causal kernel on overlapping chunks (window_attention); the native engine reads it the same."""
    s = mask.shape[-1]
    rows = torch.arange(s, device=mask.device).unsqueeze(1)
    cols = torch.arange(s, device=mask.device).unsqueeze(0)
    return mask & (cols >= (rows // window - 1) * window)


def window_attention(q, k, v, window, mask=None):
    """Causal attention within window_mask's windows: the fused kernel on chunks of two blocks (stride
    one block), each chunk's second half kept – the same result as attention over the whole text with
    window_mask, about four times faster at 8,192 tokens and a window of 1,024. With `mask` [1 or b, 1,
    s, s] (packed samples that must not see each other), each chunk takes its part of it; without, the
    plain causal kernel. q, k, v: [b, h, s, d]."""
    b, h, s, d = q.shape
    w = window
    n = -(-s // w)
    pad = n * w - s
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        if mask is not None:
            mask = F.pad(mask, (0, pad, 0, pad))
    if mask is None:
        first = attend(q[:, :, :w], k[:, :, :w], v[:, :, :w], is_causal=True)
    else:
        first = attend(q[:, :, :w], k[:, :, :w], v[:, :, :w], attn_mask=mask[:, :, :w, :w])
    if n == 1:
        return first[:, :, :s]

    def chunks(t):  # [b, h, n*w, d] -> [b * (n-1), h, 2w, d]
        return t.unfold(2, 2 * w, w).permute(0, 2, 1, 4, 3).reshape(b * (n - 1), h, 2 * w, d)

    if mask is None:
        out = attend(chunks(q), chunks(k), chunks(v), is_causal=True)
    else:
        # Each chunk's square of the mask: rows and columns (i-1)w … (i+1)w.
        m = mask.expand(b, 1, n * w, n * w)
        parts = torch.stack([m[:, :, i * w:(i + 2) * w, i * w:(i + 2) * w] for i in range(n - 1)], dim=1)
        out = attend(chunks(q), chunks(k), chunks(v), attn_mask=parts.reshape(b * (n - 1), 1, 2 * w, 2 * w))
    rest = out[:, :, w:].reshape(b, n - 1, h, w, d).permute(0, 2, 1, 3, 4).reshape(b, h, (n - 1) * w, d)
    return torch.cat([first, rest], dim=2)[:, :, :s]


def even(n):
    return max(2, int(n) // 2 * 2)


# ---- attention --------------------------------------------------------------


class FullAttention(nn.Module):
    """Causal attention with rotary positions and grouped key/value heads (Llama layout)."""

    def __init__(self, width, heads, kv_heads, p):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, width // heads
        self.window = int(p.get("window", 0) or 0)
        bias = bool(p.get("bias"))
        self.q_proj = nn.Linear(width, heads * self.dim, bias=bias)
        self.k_proj = nn.Linear(width, kv_heads * self.dim, bias=bias)
        self.v_proj = nn.Linear(width, kv_heads * self.dim, bias=bias)
        self.o_proj = nn.Linear(heads * self.dim, width, bias=False)
        self.turned = min(self.dim, even(self.dim * float(p.get("rotary", 1.0) or 1.0)))
        if p.get("qk_norm"):
            self.q_norm, self.k_norm = RMSNorm(self.dim), RMSNorm(self.dim)
        else:
            self.q_norm = self.k_norm = None

    def turn(self, x, cos, sin):
        if self.turned == self.dim:
            return rotate(x, cos, sin)
        return torch.cat([rotate(x[..., : self.turned], cos, sin), x[..., self.turned :]], dim=-1)

    def forward(self, x, cx):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = cx.rope(self.turned, torch.arange(cx.offset, cx.offset + s, device=x.device))
        q, k = self.turn(q, cos, sin), self.turn(k, cos, sin)
        if cx.cache is not None:
            return self.o_proj(self.cached(q, k, v, cx).transpose(1, 2).reshape(b, s, -1))
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        # PyTorch's fused attention: the same softmax(qk/√d)v, without keeping the whole score matrix
        # (plainly causal attention uses the fastest kernel). It exports to ONNX as plain operators.
        # (Sharing key/value heads in the kernel itself, enable_gqa, falls back to the slow one.)
        if not self.window and cx.causal:
            y = attend(q, k, v, is_causal=True)
        elif self.window and not torch.onnx.is_in_onnx_export():
            y = window_attention(q, k, v, self.window, None if cx.causal else cx.mask)
        else:
            mask = window_mask(cx.mask, self.window) if self.window else cx.mask
            y = attend(q, k, v, attn_mask=mask)
        return self.o_proj(y.transpose(1, 2).reshape(b, s, -1))


    def cached(self, q, k, v, cx):
        """Attention of the new tokens over the cached ones and themselves, adding theirs to the cache:
        writing a token at a time without reading everything before again."""
        store = cx.cache.setdefault(id(self), {})
        if "k" in store:
            k, v = torch.cat([store["k"], k], dim=2), torch.cat([store["v"], v], dim=2)
        store["k"], store["v"] = k, v
        total, s = k.shape[2], q.shape[2]
        query = torch.arange(cx.offset, cx.offset + s, device=q.device).unsqueeze(1)
        key = torch.arange(total, device=q.device).unsqueeze(0)
        allowed = key <= query
        if self.window:
            allowed = allowed & (key >= (query // self.window - 1) * self.window)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)


class LatentAttention(nn.Module):
    """Multi-head latent attention (DeepSeek-V2/V3).

    Keys and values are made from a small latent per token (all a cache would
    keep), plus a rotary key part shared by all heads; queries may come from a
    latent too.
    """

    def __init__(self, width, heads, p):
        super().__init__()
        self.heads = heads
        self.nope = int(p.get("head_dim", width // heads))
        self.rope = even(p.get("rope_dim", max(8, self.nope // 2)))
        self.v_dim = self.nope
        self.kv_rank = int(p.get("kv_rank", max(16, width // 4)))
        self.q_rank = int(p.get("q_rank", 0))
        qk = self.nope + self.rope
        if self.q_rank:
            self.q_down = nn.Linear(width, self.q_rank, bias=False)
            self.q_norm = RMSNorm(self.q_rank)
            self.q_up = nn.Linear(self.q_rank, heads * qk, bias=False)
        else:
            self.q_proj = nn.Linear(width, heads * qk, bias=False)
        self.kv_down = nn.Linear(width, self.kv_rank + self.rope, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank)
        self.kv_up = nn.Linear(self.kv_rank, heads * (self.nope + self.v_dim), bias=False)
        self.o_proj = nn.Linear(heads * self.v_dim, width, bias=False)

    def forward(self, x, cx):
        b, s, _ = x.shape
        q = self.q_up(self.q_norm(self.q_down(x))) if self.q_rank else self.q_proj(x)
        q = q.view(b, s, self.heads, self.nope + self.rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
        latent, k_rope = self.kv_down(x).split([self.kv_rank, self.rope], dim=-1)
        kv = self.kv_up(self.kv_norm(latent)).view(b, s, self.heads, self.nope + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope, self.v_dim], dim=-1)
        cos, sin = cx.rope(self.rope, torch.arange(s, device=x.device))
        q_rope = rotate(q_rope, cos, sin)
        k_rope = rotate(k_rope.unsqueeze(1), cos, sin).expand(b, self.heads, s, self.rope)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.nope + self.rope)
        y = torch.softmax(scores.masked_fill(~cx.mask, NEG), dim=-1) @ v
        return self.o_proj(y.transpose(1, 2).reshape(b, s, -1))


class Compressor(nn.Module):
    """Turns every `m` tokens into one entry: a softmax-weighted pooling, per channel.

    With `overlap`, each entry pools 2m candidates – its own block and the one
    before – as DeepSeek-V4's compressed sparse attention does.
    """

    def __init__(self, width, dim, m, overlap):
        super().__init__()
        self.m, self.overlap = m, overlap
        self.c_a = nn.Linear(width, dim, bias=False)
        self.z_a = nn.Linear(width, dim, bias=False)
        self.b_a = nn.Parameter(torch.zeros(m, dim))
        if overlap:
            self.c_b = nn.Linear(width, dim, bias=False)
            self.z_b = nn.Linear(width, dim, bias=False)
            self.b_b = nn.Parameter(torch.zeros(m, dim))

    def forward(self, x):
        b, s, _ = x.shape
        n = s // self.m
        ca = self.c_a(x).view(b, n, self.m, -1)
        za = self.z_a(x).view(b, n, self.m, -1) + self.b_a
        if not self.overlap:
            return (torch.softmax(za, dim=2) * ca).sum(2)
        cb = self.c_b(x).view(b, n, self.m, -1)
        zb = self.z_b(x).view(b, n, self.m, -1) + self.b_b
        # The block before each block (none before the first).
        cb = torch.cat([torch.zeros_like(cb[:, :1]), cb[:, :-1]], dim=1)
        zb = torch.cat([torch.full_like(zb[:, :1], NEG), zb[:, :-1]], dim=1)
        c, z = torch.cat([cb, ca], dim=2), torch.cat([zb, za], dim=2)
        return (torch.softmax(z, dim=2) * c).sum(2)


class CompressedAttention(nn.Module):
    """DeepSeek-V4's compressed sparse (CSA) and heavily compressed (HCA) attention.

    Queries come from a latent and are normalized per head; keys and values
    are one shared head (MQA): compressed entries of earlier blocks – all of
    them (HCA), or the `top_k` a lightning indexer picks (CSA) – and the last
    `window` plain tokens. A learned sink per head lets a head attend to
    little. Rotary positions cover the last `rope_dim` channels, and outputs
    are rotated back by their own position so only distances count. Heads are
    projected out in groups.
    """

    def __init__(self, width, heads, p, sparse):
        super().__init__()
        self.heads, self.sparse = heads, sparse
        self.dim = even(p.get("head_dim", width // heads))
        self.rope = min(even(p.get("rope_dim", max(8, self.dim // 4))), self.dim)
        self.m = max(1, int(p.get("compress", 4 if sparse else 32)))
        self.window = int(p.get("window", 64))
        q_rank = int(p.get("q_rank", max(16, width // 2)))
        self.q_down = nn.Linear(width, q_rank, bias=False)
        self.q_down_norm = RMSNorm(q_rank)
        self.q_up = nn.Linear(q_rank, heads * self.dim, bias=False)
        self.q_norm = RMSNorm(self.dim)
        self.compressor = Compressor(width, self.dim, self.m, overlap=sparse)
        self.kv_norm = RMSNorm(self.dim)
        self.local = nn.Linear(width, self.dim, bias=False)
        self.sink = nn.Parameter(torch.zeros(heads))
        groups = max(1, min(int(p.get("groups", 2)), heads))
        while heads % groups:
            groups -= 1
        self.groups = groups
        group_dim = int(p.get("group_dim", max(8, width // 2)))
        self.group_proj = nn.ModuleList(nn.Linear(heads // groups * self.dim, group_dim, bias=False) for _ in range(groups))
        self.o_proj = nn.Linear(groups * group_dim, width, bias=False)
        if sparse:
            self.top_k = max(1, int(p.get("top_k", 16)))
            self.ih, self.idim = int(p.get("indexer_heads", 4)), int(p.get("indexer_dim", 32))
            self.index_q = nn.Linear(q_rank, self.ih * self.idim, bias=False)
            self.index_w = nn.Linear(width, self.ih, bias=False)
            self.index_k = Compressor(width, self.idim, self.m, overlap=True)
        self.aux = None

    def _rope(self, x, positions, cx):
        cos, sin = cx.rope(self.rope, positions)
        keep, turn = x.split([self.dim - self.rope, self.rope], dim=-1)
        return torch.cat([keep, rotate(turn, cos, sin)], dim=-1)

    def forward(self, x, cx):
        b, s, _ = x.shape
        pad = (-s) % self.m
        xp = F.pad(x, (0, 0, 0, pad)) if pad else x
        n = xp.shape[1] // self.m
        positions = torch.arange(s, device=x.device)
        cq = self.q_down_norm(self.q_down(x))
        q = self.q_norm(self.q_up(cq).view(b, s, self.heads, self.dim)).transpose(1, 2)  # (b, h, s, d)
        q = self._rope(q, positions, cx)

        # Compressed entries, each placed at the last position of its block.
        entries = self.kv_norm(self.compressor(xp))  # (b, n, d)
        block_end = torch.arange(n, device=x.device) * self.m + (self.m - 1)
        entries = self._rope(entries, block_end, cx)
        # A token sees the blocks that ended before its own block began.
        eligible = torch.arange(n, device=x.device).unsqueeze(0) < (positions // self.m).unsqueeze(1)  # (s, n)
        select = eligible.unsqueeze(0).expand(b, s, n)
        index = None
        if self.sparse:
            iq = self.index_q(cq).view(b, s, self.ih, self.idim)
            ik = self.index_k(xp)  # (b, n, idim)
            w = self.index_w(x)  # (b, s, ih)
            index = (torch.relu(torch.einsum("bshd,bnd->bshn", iq, ik)) * w.unsqueeze(-1)).sum(2)  # (b, s, n)
            index = index.masked_fill(~eligible.unsqueeze(0), NEG)
            select = select & rank_mask(index, self.top_k)
        local = self._rope(self.kv_norm(self.local(x)), positions, cx)
        local_mask = window_mask(cx.mask, self.window).expand(b, 1, s, s)

        keys = torch.cat([entries, local], dim=1).unsqueeze(1)  # (b, 1, n + s, d)
        mask = torch.cat([select.unsqueeze(1), local_mask], dim=-1)  # (b, 1, s, n + s)
        logits = (q @ keys.transpose(-1, -2)) / math.sqrt(self.dim)  # (b, h, s, n + s)
        logits = logits.masked_fill(~mask, NEG)
        sink = self.sink.view(1, self.heads, 1, 1).expand(b, self.heads, s, 1)
        weights = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :-1]
        out = weights @ keys  # (b, h, s, d)
        # Rotated back by their own position: what is left depends on distance only.
        cos, sin = cx.rope(self.rope, positions)
        keep, turn = out.split([self.dim - self.rope, self.rope], dim=-1)
        out = torch.cat([keep, rotate(turn, cos, -sin)], dim=-1)

        if self.sparse and self.training and n:
            # The indexer learns to point where dense attention over the entries would look.
            with torch.no_grad():
                dense = logits[..., :n].masked_fill(~eligible.view(1, 1, s, n), NEG)
                target = torch.softmax(dense, dim=-1).mean(1)
            has = eligible.any(-1).view(1, s, 1).float()
            self.aux = 0.1 * -((target * torch.log_softmax(index, dim=-1)) * has).sum(-1).mean()

        per = self.heads // self.groups
        parts = out.transpose(1, 2).reshape(b, s, self.groups, per * self.dim)
        grouped = torch.cat([proj(parts[:, :, g]) for g, proj in enumerate(self.group_proj)], dim=-1)
        return self.o_proj(grouped)


class SSM(nn.Module):
    """A Mamba-2 style state-space layer in its quadratic (state space duality) form."""

    def __init__(self, width, heads, p):
        super().__init__()
        state = int(p.get("state", 16))
        self.heads, self.dim, self.state = heads, width // heads, state
        self.in_proj = nn.Linear(width, width * 2 + state * 2 + heads)
        self.a_log = nn.Parameter(torch.zeros(heads))
        self.y_norm = RMSNorm(width)
        self.out = nn.Linear(width, width)

    def forward(self, x, cx):
        b, s, w = x.shape
        mask = cx.mask
        z, xs, bmat, cmat, dt = torch.split(self.in_proj(x), [w, w, self.state, self.state, self.heads], dim=-1)
        dt = F.softplus(dt - 3.0).clamp(max=4.0)
        log_a = -dt * torch.exp(self.a_log)
        cum = torch.cumsum(log_a, dim=1).transpose(1, 2)
        diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)
        decay = torch.exp(diff.masked_fill(~mask, NEG))
        scores = (cmat @ bmat.transpose(-1, -2)).unsqueeze(1) / math.sqrt(self.state)
        weights = (scores * decay).masked_fill(~mask, 0.0)
        xs = (xs.view(b, s, self.heads, self.dim) * dt.unsqueeze(-1)).transpose(1, 2)
        y = (weights @ xs).transpose(1, 2).reshape(b, s, w)
        return self.out(self.y_norm(y) * F.silu(z))


# ---- feed-forward -----------------------------------------------------------


def hidden_of(width, mult):
    return max(8, int(round(width * float(mult) / 8.0)) * 8)


class SwiGLU(nn.Module):
    """A gated feed-forward layer: SiLU gates (SwiGLU), or tanh-approximated GELU ones (GeGLU, Gemma)."""

    def __init__(self, width, hidden, activation="swiglu"):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)
        self.geglu = activation == "geglu"

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh") if self.geglu else F.silu(gate)
        return self.down_proj(gate * self.up_proj(x))


class MoE(nn.Module):
    """DeepSeekMoE: shared experts for every token, and routed experts of which each uses `top_k`."""

    def __init__(self, width, p, vocab):
        super().__init__()
        hidden = hidden_of(width, p.get("mult", 1))
        self.n = max(1, int(p.get("experts", 8)))
        self.top_k = max(1, min(int(p.get("top_k", 2)), self.n))
        self.gating = p.get("gating", "softmax")
        self.balance = p.get("balance", "aux")
        self.hash = bool(p.get("hash", False))
        self.normalize = bool(p.get("normalize", self.gating != "softmax"))
        self.scale = float(p.get("scale", 1.0))
        self.shared = nn.ModuleList(SwiGLU(width, hidden) for _ in range(int(p.get("shared", 1))))
        self.experts = nn.ModuleList(SwiGLU(width, hidden) for _ in range(self.n))
        self.gate = nn.Linear(width, self.n, bias=False)
        # Auxiliary-loss-free balancing: a bias on choosing only, nudged after each step.
        self.register_buffer("bias", torch.zeros(self.n))
        if self.hash:
            # Fixed experts per token id, by a hash (DeepSeek-V4's first MoE layers).
            first = (torch.arange(vocab, dtype=torch.long) * 2654435761) % self.n
            table = torch.stack([(first + j) % self.n for j in range(self.top_k)], dim=-1)
            self.register_buffer("table", table)
        self.aux = None
        self.load = None
        self.dense = False  # every expert for every token, choices by rank (for export)
        # Grouped products need SwiGLU experts (and a PyTorch that has them): False until they fail.
        self.geglu_experts = any(e.geglu for e in self.experts)

    def grouped(self, gathered, expert_of):
        """Every routed expert on its tokens (sorted by expert) in three grouped products – one kernel
        each instead of three per expert (about three times faster), and no wait for how many tokens each
        expert took. None when PyTorch or the device cannot (then a loop over the experts)."""
        if (not gathered.is_cuda or not hasattr(torch, "_grouped_mm") or self.geglu_experts
                or torch.onnx.is_in_onnx_export()):
            return None
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else gathered.dtype
        if dtype not in (torch.bfloat16,):
            return None
        try:
            counts = torch.bincount(expert_of, minlength=self.n)
            offs = torch.cumsum(counts, 0).to(torch.int32)
            x = gathered.to(dtype)
            gate = torch.stack([e.gate_proj.weight for e in self.experts]).to(dtype).transpose(1, 2)
            up = torch.stack([e.up_proj.weight for e in self.experts]).to(dtype).transpose(1, 2)
            down = torch.stack([e.down_proj.weight for e in self.experts]).to(dtype).transpose(1, 2)
            hidden = F.silu(torch._grouped_mm(x, gate, offs=offs)) * torch._grouped_mm(x, up, offs=offs)
            return torch._grouped_mm(hidden, down, offs=offs)
        except (RuntimeError, TypeError, NotImplementedError):
            self.geglu_experts = True  # not again: the loop from now on
            return None

    def affinity(self, logits):
        if self.gating == "sigmoid":
            return torch.sigmoid(logits)
        if self.gating == "sqrt_softplus":
            # In float32 and kept off zero: a score below about -104 makes softplus 0, where the square
            # root's gradient is infinite – NaN, which spread to every weight of a long run.
            return torch.sqrt(F.softplus(logits.float()).clamp_min(1e-30))
        return torch.softmax(logits, dim=-1)

    def forward(self, x, tokens):
        b, s, w = x.shape
        y = sum((expert(x) for expert in self.shared), torch.zeros_like(x))
        affinity = self.affinity(self.gate(x))  # (b, s, n)
        if self.hash:
            chosen = torch.zeros_like(affinity).scatter(-1, self.table[tokens], 1.0) > 0
        elif self.dense:
            chosen = rank_mask(affinity + self.bias, self.top_k)
        else:
            index = (affinity + self.bias).topk(self.top_k, dim=-1).indices
            chosen = torch.zeros_like(affinity).scatter(-1, index, 1.0) > 0
        weights = affinity * chosen
        if self.normalize:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        weights = weights * self.scale
        if self.dense or not self.training:
            outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
            y = y + (weights.unsqueeze(-1) * outs).sum(-2)
        else:
            # Each token's chosen experts, sorted by expert: one wait for the graphics card per layer
            # (how many tokens each expert takes), each expert on one contiguous slice, one sum back.
            flat, wflat = x.reshape(-1, w), weights.reshape(-1, self.n)
            picked = chosen.reshape(-1, self.n)
            token, expert_of = picked.nonzero(as_tuple=True)
            order = torch.argsort(expert_of, stable=True)
            token, expert_of = token[order], expert_of[order]
            gathered = flat[token]
            outs = self.grouped(gathered, expert_of)
            if outs is None:
                counts = torch.bincount(expert_of, minlength=self.n).tolist()
                parts = [expert(part) for expert, part, c in zip(self.experts, torch.split(gathered, counts), counts) if c]
                outs = torch.cat(parts) if parts else None
            routed = torch.zeros_like(flat)
            if outs is not None:
                scale = wflat[token, expert_of].unsqueeze(-1)
                routed = routed.index_add(0, token, outs * scale.to(outs.dtype))
            y = y + routed.view(b, s, w)
        if self.training:
            share = chosen.float()
            self.load = share.sum(dim=(0, 1)).detach()
            if not self.hash:
                probs = affinity / affinity.sum(-1, keepdim=True).clamp_min(1e-9)
                if self.balance == "aux":
                    self.aux = 0.01 * self.n * (share.mean(dim=(0, 1)) * probs.mean(dim=(0, 1))).sum()
                else:
                    # A slight sequence-wise balance loss, as DeepSeek-V3 and V4 keep.
                    self.aux = 1e-4 * self.n * (share.mean(dim=1) * probs.mean(dim=1)).sum(-1).mean()
        return y


class PersonaFFN(nn.Module):
    """One feed-forward expert per persona, plus a neutral one; routing learns the persona tags."""

    def __init__(self, width, personas, p):
        super().__init__()
        hidden = hidden_of(width, p.get("mult", 2.67))
        self.experts = nn.ModuleList(SwiGLU(width, hidden) for _ in range(personas + 1))
        self.router = nn.Linear(width, personas + 1)
        self.aux = None

    def forward(self, x, labels=None):
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        outs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        if self.training and labels is not None:
            target = (labels + 1).clamp_min(0)
            self.aux = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        return (weights.unsqueeze(-1) * outs).sum(-2)


# ---- hyper-connections ------------------------------------------------------


def sinkhorn(m, iterations):
    for _ in range(iterations):
        m = m / m.sum(-2, keepdim=True).clamp_min(1e-9)
        m = m / m.sum(-1, keepdim=True).clamp_min(1e-9)
    return m


class HyperConnection(nn.Module):
    """Manifold-constrained hyper-connections (mHC, DeepSeek-V4) around one sublayer.

    The residual is n streams. A sublayer reads a sigmoid-weighted mix of
    them (A), the streams mix through a doubly stochastic matrix (B, by
    Sinkhorn–Knopp), and the sublayer's output is added back with weights C.
    All three depend on the input, and start near a plain residual connection.
    """

    def __init__(self, width, streams, iterations):
        super().__init__()
        self.n, self.iterations = streams, iterations
        flat = streams * width
        self.norm = RMSNorm(flat)
        self.w_pre = nn.Linear(flat, streams, bias=False)
        self.w_res = nn.Linear(flat, streams * streams, bias=False)
        self.w_post = nn.Linear(flat, streams, bias=False)
        for layer in (self.w_pre, self.w_res, self.w_post):
            nn.init.zeros_(layer.weight)
        self.s_pre = nn.Parameter(torch.full((streams,), -math.log(streams - 1.0) if streams > 1 else 4.0))
        self.s_res = nn.Parameter(torch.eye(streams) * 6.0)
        self.s_post = nn.Parameter(torch.zeros(streams))
        self.a_pre = nn.Parameter(torch.tensor(0.01))
        self.a_res = nn.Parameter(torch.tensor(0.01))
        self.a_post = nn.Parameter(torch.tensor(0.01))

    def forward(self, streams, sublayer):
        b, s, n, w = streams.shape
        h = self.norm(streams.reshape(b, s, n * w))
        a = torch.sigmoid(self.a_pre * self.w_pre(h) + self.s_pre)  # (b, s, n)
        mix = sinkhorn(torch.exp(self.a_res * self.w_res(h).view(b, s, n, n) + self.s_res), self.iterations)
        c = 2 * torch.sigmoid(self.a_post * self.w_post(h) + self.s_post)
        y = sublayer((a.unsqueeze(-1) * streams).sum(2))  # (b, s, w)
        return mix @ streams + c.unsqueeze(-1) * y.unsqueeze(2)


# ---- layers -----------------------------------------------------------------


class Layer(nn.Module):
    """A residual layer: a mixer (attention or SSM), then a feed-forward part, each pre-normed."""

    def __init__(self, width, heads, kv_heads, spec, personas, vocab, streams=0, iterations=20):
        super().__init__()
        attention, ffn = spec.get("attention", {}) or {}, spec.get("ffn", {}) or {}
        kind = attention.get("kind", "full")
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(width), RMSNorm(width)
        self.sandwich = spec.get("norm") == "sandwich"
        if self.sandwich:
            self.attn_out_norm, self.mlp_out_norm = RMSNorm(width), RMSNorm(width)
        if kind == "ssm":
            self.self_attn = SSM(width, heads, attention)
        elif kind == "mla":
            self.self_attn = LatentAttention(width, heads, attention)
        elif kind in ("csa", "hca"):
            self.self_attn = CompressedAttention(width, heads, attention, sparse=kind == "csa")
        else:
            self.self_attn = FullAttention(width, heads, kv_heads, attention)
        fkind = ffn.get("kind", "dense")
        if fkind == "moe":
            self.mlp = MoE(width, ffn, vocab)
        elif fkind == "persona":
            self.mlp = PersonaFFN(width, personas, ffn)
        else:
            self.mlp = SwiGLU(width, hidden_of(width, ffn.get("mult", 2.67)), ffn.get("activation", "swiglu"))
        self.hyper = streams > 1
        if self.hyper:
            self.hc_attn = HyperConnection(width, streams, iterations)
            self.hc_mlp = HyperConnection(width, streams, iterations)

    def _attn(self, x, cx):
        out = self.self_attn(self.input_layernorm(x), cx)
        return self.attn_out_norm(out) if self.sandwich else out

    def _mlp(self, x, cx):
        h = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoE):
            out = self.mlp(h, cx.tokens)
        elif isinstance(self.mlp, PersonaFFN):
            out = self.mlp(h, cx.labels)
        else:
            out = self.mlp(h)
        return self.mlp_out_norm(out) if self.sandwich else out

    def forward(self, x, cx):
        if self.hyper:
            x = self.hc_attn(x, lambda h: self._attn(h, cx))
            return self.hc_mlp(x, lambda h: self._mlp(h, cx))
        x = x + self._attn(x, cx)
        return x + self._mlp(x, cx)


class TextModel(nn.Module):
    """Tokens in, a score per possible next token out."""

    def __init__(self, vocab, architecture, prefix_id=None, personas=0):
        super().__init__()
        a = architecture
        width, heads = int(a.get("width", 128)), int(a.get("heads", 4))
        kv_heads = int(a.get("kv_heads", heads) or heads)
        if kv_heads <= 0 or heads % kv_heads:
            kv_heads = heads
        self.context = int(a.get("context", 256))
        self.theta = float(a.get("rope_theta", 10000.0))
        self.prefix_split = bool(a.get("prefix_split"))
        self.prefix_id = prefix_id
        hc = a.get("hyper_connections") or {}
        self.streams = int(hc.get("streams", 0)) if hc else 0
        iterations = int(hc.get("sinkhorn", 20)) if hc else 20
        self.embed_tokens = nn.Embedding(vocab, width)
        self.tied = bool(a.get("tied"))
        if self.tied:
            # Also the output head: small, as GPT-2's, so the first scores are near even rather than
            # spread by sqrt(width) – the first hundred steps learn instead of shrinking them.
            nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.embed_scale = math.sqrt(width) if a.get("embed_scale") else 1.0
        self.learned_positions = a.get("positions", "rope") == "learned"
        if self.learned_positions:
            self.position = nn.Embedding(self.context, width)
        specs = a.get("layers") or [{"attention": {"kind": "full"}, "ffn": {"kind": "dense"}}] * 2
        self.layers = nn.ModuleList(
            Layer(width, heads, kv_heads, spec, personas, vocab, self.streams, iterations) for spec in specs
        )
        self.norm = RMSNorm(width)
        if not self.tied:
            self.lm_head = nn.Linear(width, vocab, bias=False)
        mtp = a.get("mtp") or {}
        self.mtp_weight = float(mtp.get("weight", 0.3)) if mtp else 0.0
        if mtp:
            # Multi-token prediction (DeepSeek-V3/V4): one more layer predicts the token after next.
            self.mtp_hidden_norm = RMSNorm(width)
            self.mtp_embed_norm = RMSNorm(width)
            self.mtp_proj = nn.Linear(2 * width, width, bias=False)
            self.mtp_layer = Layer(width, heads, kv_heads, {}, personas, vocab)
        self.checkpointing = False

    def can_cache(self):
        """Whether it can write with a cache: plain attention layers (with or without windows), rotary
        positions, one residual stream."""
        return (self.streams <= 1 and not self.learned_positions and not self.prefix_split
                and all(isinstance(layer.self_attn, FullAttention) for layer in self.layers))

    def head(self, h):
        """Scores per token: the output head, or the embedding table when they are tied."""
        return F.linear(h, self.embed_tokens.weight) if self.tied else self.lm_head(h)

    def token_loss(self, hidden, targets):
        """The mean cross-entropy of the head over the positions with a target (not -100), without
        keeping the scores of them all (`SlicedLoss`)."""
        h, t = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        keep = t != -100
        h, t = h[keep], t[keep]
        if not t.numel():
            return hidden.sum() * 0.0
        weight = self.embed_tokens.weight if self.tied else self.lm_head.weight
        kind = h.device.type
        dtype = torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else h.dtype
        return SlicedLoss.apply(h, weight, t, dtype)

    def embed(self, tokens):
        x = self.embed_tokens(tokens)
        return x * self.embed_scale if self.embed_scale != 1.0 else x

    def base_mask(self, tokens, segments=None):
        b, s = tokens.shape
        positions = torch.arange(s, device=tokens.device)
        mask = (positions.unsqueeze(0) <= positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        if segments is not None:
            # Samples packed into one row only see themselves.
            mask = mask & (segments.unsqueeze(-1) == segments.unsqueeze(-2)).unsqueeze(1)
        if self.prefix_split and self.prefix_id is not None:
            marks = (tokens == self.prefix_id).long() * (positions + 1).unsqueeze(0)
            prefix = marks.max(dim=1).values.view(b, 1, 1, 1)
            inside = (positions.view(1, 1, s, 1) < prefix) & (positions.view(1, 1, 1, s) < prefix)
            mask = mask | inside
        return mask

    def forward(self, tokens, labels=None, targets=None, segments=None, loss=False, cache=None):
        """Logits for every position – or with `loss` and targets, the loss alone, which needs far
        less memory; when training, also the auxiliary loss (balancing, the indexer, multi-token
        prediction) and the experts' loads."""
        for module in self.modules():
            if hasattr(module, "aux"):
                module.aux = None
            if isinstance(module, MoE):
                module.load = None
        s = tokens.shape[1]
        x = self.embed(tokens)
        if self.learned_positions:
            x = x + self.position(torch.arange(s, device=tokens.device)).unsqueeze(0)
        causal = not (self.prefix_split and self.prefix_id is not None) and segments is None
        cx = Context(tokens, self.base_mask(tokens, segments), self.theta, labels, not self.learned_positions, causal,
                     cache)
        if self.streams > 1:
            x = x.unsqueeze(2).expand(-1, -1, self.streams, -1).contiguous()
        for layer in self.layers:
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(lambda h, layer=layer: layer(h, cx), x, use_reentrant=False)
            else:
                x = layer(x, cx)
        if self.streams > 1:
            x = x.mean(2)
        out = {}
        if loss and targets is not None:
            out["loss"] = self.token_loss(self.norm(x), targets).reshape(1)
        elif cache is not None:
            # Writing: the scores of the last token only, and the cache moves on.
            cache["length"] = cx.offset + s
            out["logits"] = self.head(self.norm(x[:, -1:]))
        else:
            out["logits"] = self.head(self.norm(x))
        aux = torch.zeros((), device=tokens.device)
        for module in self.modules():
            if getattr(module, "aux", None) is not None:
                aux = aux + module.aux
        if self.training and self.mtp_weight and targets is not None and s > 1:
            nxt = self.embed(tokens[:, 1:])
            h = self.mtp_proj(torch.cat([self.mtp_hidden_norm(x[:, :-1]), self.mtp_embed_norm(nxt)], dim=-1))
            cx2 = Context(tokens[:, 1:], cx.mask[..., : s - 1, : s - 1], self.theta, None, not self.learned_positions, causal)
            later = self.norm(self.mtp_layer(h, cx2))
            aux = aux + self.mtp_weight * self.token_loss(later, targets[:, 1:])
        loads = [m.load for m in self.modules() if isinstance(m, MoE) and m.load is not None]
        load = torch.cat(loads).unsqueeze(0) if loads else torch.zeros(1, 0, device=tokens.device)
        return dict(out, aux=aux.reshape(1), load=load)

    def balance(self, load, rate=1e-3):
        """Auxiliary-loss-free balancing: experts chosen more than average get a lower bias."""
        offset = 0
        for module in self.modules():
            if isinstance(module, MoE) and not module.hash:
                part = load[offset : offset + module.n] if load.numel() >= offset + module.n else None
                offset += module.n
                if part is not None and module.balance == "bias" and part.sum() > 0:
                    module.bias -= rate * torch.sign(part - part.mean())
            elif isinstance(module, MoE):
                offset += module.n


class Exported(nn.Module):
    """tokens -> logits, for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        for module in model.modules():
            if isinstance(module, MoE):
                module.dense = True

    def forward(self, tokens):
        return self.model(tokens)["logits"]


class VisionModel(nn.Module):
    """A picture in, a score per class out: `input` [1, 3, h, w] -> `logits` [1, classes]."""

    def __init__(self, classes, architecture):
        super().__init__()
        p = architecture.get("conv") or {}
        layers, channels = int(p.get("layers", 3)), int(p.get("channels", 32))
        parts, c_in = [], 3
        for i in range(max(1, layers)):
            c_out = channels * (2 ** min(i, 3))
            parts += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*parts)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(c_in, classes)

    def forward(self, x):
        return self.classify(self.pool(self.features(x)).flatten(1))


# ---- adapters: LoRA and DoRA ------------------------------------------------


class Adapted(nn.Module):
    """A frozen linear layer with a low-rank update; DoRA also learns each output's magnitude."""

    def __init__(self, base, rank, alpha, dora=False):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        device = base.weight.device
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        self.scale = alpha / rank
        self.dora = dora
        if dora:
            self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))

    def weight(self):
        w = self.base.weight + self.scale * (self.b @ self.a)
        if self.dora:
            w = self.magnitude.unsqueeze(1) * w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return w

    def forward(self, x):
        return F.linear(x, self.weight(), self.base.bias)

    def merged(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged = merged.to(self.base.weight.device)
        with torch.no_grad():
            merged.weight.copy_(self.weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


# Linear layers adapters go on: attention and feed-forward weights, not routers or mixing weights.
ADAPTED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "q_up", "q_down", "kv_up", "kv_down", "local", "in_proj", "out")


def add_adapters(model, rank=8, alpha=16, dora=False):
    """Freezes the model and puts adapters on the linear layers of its layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.layers.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in ADAPTED:
                setattr(module, name, Adapted(child, rank, alpha, dora))


def merge_adapters(model):
    """Folds adapters back into plain linear layers, so the model is as before, updated."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Adapted):
                setattr(module, name, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
