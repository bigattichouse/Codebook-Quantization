# Histogram-Guided Structural Neuron Pruning

A two-stage approach to identifying pruning candidates using the existing
codebook histogram infrastructure, extended with downstream path analysis.
The goal is a principled, data-free structural ranking of neurons before
any functional (inference-based) confirmation is performed.

---

## Core Idea

The codebook compression work demonstrated that trained model weight
distributions are highly non-uniform — the actual vocabulary of values used
is far smaller than the storage format implies. The same histogram tooling
that reveals value-distribution structure can be redirected to ask a different
question: which neurons are structurally expendable?

Standard magnitude pruning picks an arbitrary percentile threshold and accepts
quality loss as inevitable. This approach uses two independent structural
metrics that measure different things — local footprint and downstream reach
— so that only neurons where both metrics agree are treated as strong
pruning candidates. This is more principled than a single threshold but still
not a guarantee of zero quality loss; functional confirmation via inference
is always the final step.

---

## Concepts and Definitions

### Neuron

In a Linear layer with weight matrix W of shape `[out_features, in_features]`,
each **row** is an output neuron. Row i computes:

```
output[i] = sum_j(input[j] * W[i, j])
```

The L2 norm of row i — `||W[i, :]||_2` — is the neuron's **magnitude score**:
how large its weights are in aggregate. A magnitude score near zero means the
neuron produces near-zero output regardless of input.

### Pruning a Neuron

Removing neuron i from layer L requires a coordinated change across two
weight matrices:

1. Remove **row i** from `W_L` — the output projection of layer L
   (shape changes from `[out, in]` to `[out-1, in]`)
2. Remove **column i** from `W_{L+1}` — the input projection of layer L+1
   (shape changes from `[out2, out]` to `[out2, out-1]`)

These must be done together. The network must remain dimensionally consistent.
In transformer architectures the residual stream adds complexity (see below).

### Downstream Path

A **path** from neuron i in layer L to neuron k in layer L+1 is the direct
connection: neuron i's output is one of the inputs to neuron k.

The **depth-1 path score** for the path i→k is:

```
score(i→k) = |W_{L+1}[k, i]|
```

i.e., the absolute value of the weight in the next layer's matrix that
connects neuron i's output to neuron k's input.

A **depth-2 path score** for i→j→k extends one layer further:

```
score(i→j→k) = |W_{L+1}[j, i]| * |W_{L+2}[k, j]|
```

The full set of depth-D path scores for candidate neuron i forms a
distribution — one score per terminal neuron at depth D. This distribution
is what we build a histogram over and use to compute SNR.

**Only downstream paths matter.** The upstream context (what inputs activated
the neuron) is irrelevant to the question of how much damage its removal
causes. Importance is entirely a function of what the neuron's output
connects to and how strongly that propagates forward.

### Signal-to-Noise Ratio (SNR)

Given the histogram of path scores for a candidate neuron, define a threshold
T separating "signal" paths (strong downstream contribution) from "noise"
paths (weak, near-zero contribution):

```
signal_power = sum(score^2  for score in path_scores  if score >= T)
noise_power  = sum(score^2  for score in path_scores  if score <  T)

SNR (dB) = 10 * log10(signal_power / noise_power)
```

A neuron with SNR of -30 dB has 1000:1 noise-to-signal power ratio — almost
everything it contributes is diffuse and near-zero downstream. A neuron at
0 dB has equal signal and noise — ambiguous.

The dB scale is appropriate here because path scores are products of weights
and span several orders of magnitude. A linear scale compresses the
interesting variation into an unreadable range.

---

## Stage 1: Histogram Filter

### Purpose

Cheap, whole-model first pass. Computes per-neuron magnitude scores across
all weight matrices, builds a histogram of those scores per layer, and flags
the low-magnitude tail as candidates for Stage 2 analysis.

### Pseudocode

```python
def stage1_histogram_filter(model, threshold_percentile=10):
    """
    Scan all linear weight matrices in the model.
    Return list of NeuronCandidate objects for neurons below the threshold.

    threshold_percentile: bottom N% of neurons per layer are flagged.
    If the norm histogram shows a natural gap, use the gap instead.
    """
    candidates = []

    for layer_name, weight_matrix in get_linear_layers(model):
        # weight_matrix shape: [out_features, in_features]

        # Compute L2 norm of each output neuron (row)
        norms = []
        for i in range(weight_matrix.shape[0]):
            norm = L2_norm(weight_matrix[i, :])
            norms.append(norm)

        # Build histogram of norms for this layer
        hist, bin_edges = build_histogram(norms, n_bins=100)

        # Find threshold: natural gap or percentile fallback
        threshold = find_natural_gap(hist, bin_edges)
        if threshold is None:
            threshold = percentile(norms, threshold_percentile)

        # Flag neurons below threshold
        for i, norm in enumerate(norms):
            if norm < threshold:
                candidates.append(NeuronCandidate(
                    layer_name=layer_name,
                    neuron_index=i,
                    magnitude_score=norm,
                    layer_norms=norms,           # full distribution for context
                    threshold_used=threshold,
                ))

    return candidates


def find_natural_gap(hist, bin_edges, min_gap_ratio=0.5):
    """
    Look for a bimodal distribution with a clear valley between two peaks.
    Returns the bin edge at the lowest point of the valley if found,
    None if the distribution is unimodal.

    min_gap_ratio: valley must be at most this fraction of the lower peak's
    height to count as a natural gap.
    """
    # Find local minima between the first and last nonzero bins
    # If the minimum between two peaks is < min_gap_ratio * lower_peak,
    # treat it as a natural gap and return the corresponding bin edge.
    ...
    return gap_threshold_or_None
```

### What to Skip

Not all weight matrices are appropriate for neuron pruning:

- **Embedding layers** (`embed_tokens`): rows are token representations, not
  neurons in the same sense. Skip or handle separately.
- **LM head** (output projection to vocabulary): directly controls output
  logits. Do not prune.
- **LayerNorm** (scale/bias parameters): different character from linear
  weights. Skip entirely.
- **Very small layers** (fewer than ~64 neurons): the statistical basis for
  histogram analysis is weak. Skip.
- **Bias terms**: handle separately from weight matrices; not prunable the
  same way.

### Transformer-Specific Layer Mapping

In a standard transformer (e.g., Qwen3):

| Module | Weight shape | Row = ? |
|---|---|---|
| `q_proj` | `[num_heads * head_dim, hidden]` | One query neuron |
| `k_proj` | `[num_kv_heads * head_dim, hidden]` | One key neuron |
| `v_proj` | `[num_kv_heads * head_dim, hidden]` | One value neuron |
| `o_proj` | `[hidden, num_heads * head_dim]` | One hidden output neuron |
| `gate_proj` | `[intermediate, hidden]` | One intermediate neuron |
| `up_proj` | `[intermediate, hidden]` | One intermediate neuron |
| `down_proj` | `[hidden, intermediate]` | One hidden output neuron |

**Attention head pruning** is a special case: rather than pruning individual
neurons within a head, prune entire heads (blocks of `head_dim` contiguous
rows in Q/K/V and corresponding columns in O). Compute per-head scores by
aggregating over the head's rows. This is structurally cleaner than
single-neuron pruning in attention layers.

For an initial implementation, focus on **MLP layers** (`gate_proj`,
`up_proj`, `down_proj`). The path structure is simpler (no attention mixing)
and the results will be more interpretable.

---

## Stage 2: Downstream Path Analysis

### Purpose

For each candidate from Stage 1, trace forward through the network to build
a distribution of downstream path scores. Compute SNR from that distribution.
This distinguishes between neurons that are small-and-isolated (safe to
prune) from neurons that are small-but-load-bearing (keep despite low magnitude).

### Pseudocode

```python
def stage2_path_analysis(model, candidates, depth=2, snr_threshold_db=-20.0):
    """
    For each candidate neuron, trace downstream paths up to `depth` layers.
    Compute the histogram of path scores and derive SNR.

    depth: how many layers forward to trace. 2-3 is usually sufficient;
           deeper tracing has diminishing returns and higher cost.

    snr_threshold_db: candidates with SNR below this are strong prune targets.
    """
    results = []

    for candidate in candidates:
        path_scores = compute_path_scores(
            model,
            layer_name=candidate.layer_name,
            neuron_index=candidate.neuron_index,
            depth=depth,
        )

        snr_db = compute_snr(path_scores)

        results.append(NeuronAnalysis(
            candidate=candidate,
            path_scores=path_scores,
            snr_db=snr_db,
            recommendation=classify(candidate.magnitude_score, snr_db),
        ))

    return sorted(results, key=lambda r: (r.snr_db, r.candidate.magnitude_score))


def compute_path_scores(model, layer_name, neuron_index, depth):
    """
    Trace all downstream paths from neuron `neuron_index` in `layer_name`
    to depth `depth` and return a flat list of terminal path scores.

    Each path score is the product of absolute weight values along the path.
    """
    # Get the column in the *next* layer's weight matrix that corresponds
    # to this neuron's output — this is the set of depth-1 connections.

    next_layer = get_downstream_layer(model, layer_name)
    if next_layer is None:
        return []  # No downstream layer (e.g., final projection)

    # W_next shape: [out_next, out_current]
    # Column neuron_index gives weights connecting this neuron to all
    # neurons in the next layer.
    depth1_scores = abs(next_layer.weight[:, neuron_index])
    # depth1_scores shape: [out_next] — one score per next-layer neuron

    if depth == 1:
        return depth1_scores.tolist()

    # Recurse: for each next-layer neuron j, trace its downstream paths
    # weighted by depth1_scores[j].
    all_scores = []
    for j, d1_score in enumerate(depth1_scores):
        if d1_score < EPSILON:
            continue  # Path is already near zero; don't recurse

        deeper_scores = compute_path_scores(
            model,
            layer_name=next_layer.name,
            neuron_index=j,
            depth=depth - 1,
        )
        # Weight deeper scores by this path's contribution
        all_scores.extend(s * d1_score for s in deeper_scores)

    return all_scores


def compute_snr(path_scores, signal_threshold=None):
    """
    Compute SNR in dB from a list of path scores.

    signal_threshold: absolute value separating signal from noise paths.
    If None, use the median as the threshold (50/50 split).
    If the histogram shows a natural gap, use that instead.
    """
    if not path_scores or all(s == 0 for s in path_scores):
        return float('-inf')

    if signal_threshold is None:
        signal_threshold = median(path_scores)

    signal_power = sum(s**2 for s in path_scores if s >= signal_threshold)
    noise_power  = sum(s**2 for s in path_scores if s <  signal_threshold)

    if noise_power < EPSILON:
        return float('+inf')  # All paths are signal — definitely keep
    if signal_power < EPSILON:
        return float('-inf')  # All paths are noise — strong prune candidate

    return 10 * math.log10(signal_power / noise_power)


def classify(magnitude_score, snr_db):
    """
    Combine magnitude score and SNR into a pruning recommendation.
    Thresholds here are starting points and should be tuned empirically.
    """
    low_magnitude = magnitude_score < MAGNITUDE_THRESHOLD
    low_snr = snr_db < SNR_THRESHOLD_DB  # e.g., -20 dB

    if low_magnitude and low_snr:
        return "prune_candidate"      # Both metrics agree: structurally dead
    elif low_magnitude and not low_snr:
        return "caution"              # Quiet locally but downstream reach is real
    elif not low_magnitude and low_snr:
        return "review"               # Active locally, washed out downstream
    else:
        return "keep"                 # Both metrics say it matters
```

### Efficiency Considerations

The `compute_path_scores` recursion has branching factor equal to the next
layer's output dimension (typically 2048–8192 in a 1.7B model). At depth 2
the number of terminal paths could be millions. Pruning the recursion
aggressively is necessary:

```python
# Prune branches whose contribution is already negligible
if d1_score < EPSILON:
    continue

# Or: only recurse into the top-K next-layer neurons by depth-1 score
top_k = 64  # tunable
top_indices = argsort(depth1_scores)[-top_k:]
for j in top_indices:
    ...
```

For the SNR computation we don't need all paths — we need the distribution
of path scores. Sampling a few thousand paths is sufficient to characterize
the distribution accurately.

---

## Decision Matrix

```
                    SNR high (> threshold)   SNR low (< threshold)
                   ┌─────────────────────────┬────────────────────────────┐
Magnitude high     │  KEEP                   │  REVIEW                    │
                   │  Active locally,        │  Large locally but output  │
                   │  reaches output well.   │  washed out downstream.    │
                   ├─────────────────────────┼────────────────────────────┤
Magnitude low      │  CAUTION                │  PRUNE CANDIDATE           │
                   │  Small locally but      │  Both metrics agree:       │
                   │  downstream path is     │  structurally inert.       │
                   │  strong. Don't prune    │  Forward to functional     │
                   │  on magnitude alone.    │  confirmation.             │
                   └─────────────────────────┴────────────────────────────┘
```

Prune with confidence from the bottom-right quadrant only. The bottom-left
quadrant is the most important insight of the two-metric approach: neurons
that naive magnitude pruning would flag but should be left alone.

---

## Functional Confirmation (Inference Required)

**Structural analysis cannot determine functional importance.** It measures
weight topology — size and downstream reach — but cannot determine whether
a neuron is unique in what it contributes, or whether another neuron already
covers the same function for typical inputs.

The only way to assess functional importance is ablation:

```python
def ablation_test(model, candidate, calibration_set, divergence_threshold=0.01):
    """
    Zero out candidate neuron in the model, run calibration set,
    measure output divergence vs. the original model.

    Returns True if the neuron is confirmed safe to prune (low divergence).
    """
    original_outputs = run_inference(model, calibration_set)

    # Zero out the neuron (non-destructive test before permanent pruning)
    with temporary_ablation(model, candidate):
        ablated_outputs = run_inference(model, calibration_set)

    divergence = mean_kl_divergence(original_outputs, ablated_outputs)
    # Alternatively: cosine similarity of final hidden states, or
    # exact token match rate on greedy decoding.

    return divergence < divergence_threshold


def temporary_ablation(model, candidate):
    """Context manager: zero out neuron during the with-block, restore after."""
    layer = get_layer(model, candidate.layer_name)
    saved = layer.weight[candidate.neuron_index, :].clone()
    layer.weight[candidate.neuron_index, :] = 0
    yield
    layer.weight[candidate.neuron_index, :] = saved
```

A calibration set of 200–500 representative examples is usually sufficient
to catch meaningful quality degradation. Full benchmark evaluation is the
thorough version; ablation on a calibration set is the fast version.

### Recommended Workflow

```
1. Stage 1: Histogram filter
   → ranked candidate list (whole model, fast, data-free)

2. Stage 2: Path analysis on candidates only
   → SNR per candidate (targeted, data-free)

3. Decision matrix: flag "prune_candidate" neurons
   → structured list, ranked by combined score

4. Ablation on calibration set (inference required)
   → confirm candidates with negligible functional impact

5. Permanent pruning of confirmed candidates
   → remove rows/columns, save modified model

6. Benchmark on downstream task
   → measure actual quality change

7. Iterate conservatively (small batches, not all candidates at once)
```

---

## Data Structures

```python
@dataclass
class NeuronCandidate:
    layer_name: str          # e.g., "model.layers.12.mlp.gate_proj"
    layer_index: int         # transformer layer number
    neuron_index: int        # row index in weight matrix
    magnitude_score: float   # L2 norm of the row
    layer_norms: List[float] # all norms in this layer (for context/percentile)
    threshold_used: float    # threshold that flagged this candidate


@dataclass
class NeuronAnalysis:
    candidate: NeuronCandidate
    path_scores: List[float]  # sampled downstream path scores
    signal_power: float
    noise_power: float
    snr_db: float
    recommendation: str       # "prune_candidate", "caution", "review", "keep"


@dataclass
class PruningReport:
    total_neurons_scanned: int
    stage1_candidates: int
    stage2_prune_candidates: int
    stage2_caution: int
    analyses: List[NeuronAnalysis]  # sorted by pruning confidence

    def summary(self):
        return (
            f"Scanned {self.total_neurons_scanned} neurons. "
            f"Stage 1 flagged {self.stage1_candidates} candidates "
            f"({100*self.stage1_candidates/self.total_neurons_scanned:.1f}%). "
            f"Stage 2 confirmed {self.stage2_prune_candidates} prune candidates, "
            f"{self.stage2_caution} caution."
        )
```

---

## Applying Permanent Pruning

Once functional confirmation is complete:

```python
def prune_neuron(model, candidate):
    """
    Permanently remove a neuron from the model.
    Modifies the weight matrices of both the layer containing the neuron
    and the immediately downstream layer.
    """
    layer = get_layer(model, candidate.layer_name)
    next_layer = get_downstream_layer(model, candidate.layer_name)

    i = candidate.neuron_index

    # Remove row i from current layer (output dimension shrinks by 1)
    new_weight = torch.cat([
        layer.weight[:i, :],
        layer.weight[i+1:, :]
    ], dim=0)
    layer.weight = nn.Parameter(new_weight)

    if layer.bias is not None:
        new_bias = torch.cat([layer.bias[:i], layer.bias[i+1:]])
        layer.bias = nn.Parameter(new_bias)

    # Remove column i from downstream layer (input dimension shrinks by 1)
    if next_layer is not None:
        new_weight = torch.cat([
            next_layer.weight[:, :i],
            next_layer.weight[:, i+1:]
        ], dim=1)
        next_layer.weight = nn.Parameter(new_weight)

    # Note: prune in batches (all candidates in a layer at once) to avoid
    # index shifting errors. Sort candidate indices in descending order
    # before pruning so earlier removals don't shift later indices.
```

**Important**: when pruning multiple neurons from the same layer, sort
indices in descending order and prune from highest to lowest. Removing row 5
shifts row 6 to index 5, row 7 to index 6, etc. — pruning descending avoids
this index corruption.

---

## Known Limitations

### Structural ≠ Functional

Both metrics operate on weight matrices only, with no input data.
Weight magnitude and downstream path scores tell you about structural
topology, not functional behavior. A neuron could score as a strong prune
candidate structurally and still be load-bearing for specific inputs that
aren't well-represented in the calibration set.

**This is why functional confirmation is not optional.** It is the step
that closes the gap between "structurally plausible to prune" and
"confirmed safe to prune."

### Residual Connections

Transformer architectures use residual connections: each layer's output
is added to the residual stream rather than replacing it. This has two
implications:

1. A neuron that produces near-zero output in its layer may still have
   near-zero effect on the network (the residual bypass carries most of
   the information). This makes magnitude pruning somewhat safer in
   transformers than in non-residual architectures.

2. The "downstream layer" for path analysis is the immediately following
   linear projection within the same block, not necessarily the next
   transformer layer. Get the layer graph right before tracing paths.

### Attention vs. MLP

In MLP layers, paths are straightforward linear chains. In attention layers,
the attention mechanism mixes information across positions before the
projection, meaning a single "path" through an attention neuron branches
across all sequence positions. The path score computation is cleaner and
more interpretable for MLP neurons. **Start with MLP layers.**

### LoRA Adapters

If the model has LoRA adapters applied (as in the WorldModel fine-tuning),
the effective weight for a LoRA-targeted layer is:

```
W_effective = W_base + (lora_B @ lora_A) * scaling
```

The analysis should be run on `W_effective`, not `W_base` alone, to correctly
account for the fine-tuning delta. The base model weights and LoRA delta must
be merged before running the histogram analysis.

---

## Relationship to Existing Codebook Infrastructure

The Stage 1 histogram filter is a direct extension of the existing tooling:

| Existing | Extension needed |
|---|---|
| `analyze.py` — per-tensor value histograms | Change to per-row norm histogram |
| `survey.py` — model-wide survey | Add per-layer norm distribution report |
| `src/analyze_tensor.py` — tensor statistics | Add `row_norms()` and `norm_histogram()` helpers |
| `src/adaptive_compressor.py` — 100% weight scan | Reuse scan loop; redirect to norm computation |

Stage 2 (path scoring) is new infrastructure. Suggested new files:

```
src/neuron_analysis.py     — Stage 1: NeuronCandidate, histogram filter
src/path_analysis.py       — Stage 2: path tracing, SNR computation
prune.py                   — main script: runs pipeline, reports, optionally prunes
```

---

## Open Empirical Questions

Before implementing the full pipeline, the following should be checked first
(fast, requires only running the histogram analysis on a real model):

1. **Is the norm histogram bimodal?** If the distribution of per-neuron norms
   is unimodal (a smooth bell curve), the natural-gap criterion doesn't apply
   and thresholds are arbitrary. This is the most important question — look at
   a few layers of the target model before building more infrastructure.

2. **How deep does path tracing need to go?** Does the SNR converge after
   depth 1, or does it change significantly at depth 2 and 3? Depth 1 is
   much cheaper; if it's sufficient, skip the recursion.

3. **What fraction of neurons are in the prune-candidate quadrant?**
   If it's < 1%, the method has limited practical impact. If it's > 20%,
   the assumptions may be too aggressive and caution is warranted.

4. **Does ablation confirm the structural prediction?** The most important
   validation: take the top structural candidates, ablate them, measure
   divergence. If structural ranking correlates with low ablation divergence,
   the method is working. If not, the structural metrics aren't predictive
   enough to be useful.
