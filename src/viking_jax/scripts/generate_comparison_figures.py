#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates 3 summary PNG figures comparing SHACA, Viking, and SEANet in DAL.

Produces:
  1. `fig1_architecture_and_metrics_overview.png`:
     Latency vs. parameter Pareto frontier + CARFAC auditory losses + scorecard.
  2. `fig2_cochlear_nap_heatmaps.png`:
     Healthy vs. Impaired vs. Aided CARFAC Neural Activity Pattern (NAP)
     cochleograms on real speech/audio (`binaural_test.wav` + noise).
  3. `fig3_smoke_vs_real_wav_training.png`:
     5-step synthetic smoke run vs. 30-step real-WAV (`long_test.wav` /
     `binaural_test.wav`) training convergence + HASPI v2 / HASQI v2 / STOI.
"""

import json
import os
import tempfile
from collections.abc import Mapping
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import matplotlib
from absl import app
from absl import flags
from hp_acoustic.data import clips as data_clips
from hp_acoustic.data import pipeline
from hp_acoustic.data import records
from hp_acoustic.data import transforms
from hp_acoustic.data.sources import color_noise
from hp_acoustic.data.sources import processed
from hp_acoustic.losses import representation_losses
from hp_acoustic.losses import terms as terms_lib
from hp_acoustic.losses import waveform_losses
from hp_acoustic.metrics import context as metric_context
from hp_acoustic.metrics import perceptual
from hp_acoustic.models import audio_model
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea
from hp_acoustic.models import context as auditory_context
from hp_acoustic.training import data as data_lib
from hp_acoustic.training import loop
from hp_acoustic.training import steps

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import shaca_jax
import soundfile as sf
from scipy.signal import resample_poly
from seanet.models import seanet

from viking_jax import viking_jax

_OUTPUT_DIR = flags.DEFINE_string(
  "output_dir",
  "/Users/shaikath/Downloads/viking-main/figures",
  "Directory where PNG figures and benchmark_summary.json are saved.",
)
_SAMPLE_RATE = flags.DEFINE_integer(
  "sample_rate", 16000, "Target sample rate in Hz."
)
_SMOKE_STEPS = flags.DEFINE_integer(
  "smoke_steps", 5, "Optimization steps for the quick synthetic smoke run."
)
_REAL_STEPS = flags.DEFINE_integer(
  "real_steps", 30, "Optimization steps for the real-WAV training run."
)
_EVAL_EVERY = flags.DEFINE_integer(
  "eval_every", 5, "Validation interval in steps for convergence curves."
)
_OHC_HEALTH = flags.DEFINE_float(
  "impaired_ohc_health", 0.5, "Impaired CARFAC outer-hair-cell health."
)

# Colorblind-safe palette (Okabe-Ito inspired)
_COLORS = {
  "Unprocessed (Impaired)": "#7F8C8D",
  "SHACA (Prescribed)": "#E69F00",
  "SHACA (DAL-Trained)": "#D55E00",
  "Viking (Min-Phase)": "#0072B2",
  "Viking (Linear-Phase)": "#56B4E9",
  "SEANet (Causal Conv)": "#009E73",
}


class _IdentityAudioModel:
  """Unprocessed noisy pass-through baseline."""

  @property
  def name(self) -> str:
    return "unprocessed"

  def init(self, rng: jax.Array) -> dict[str, jax.Array]:
    del rng
    return {}

  def apply(
    self, params: Mapping[str, jax.Array], audio: jnp.ndarray
  ) -> jnp.ndarray:
    del params
    return audio

  def constrain(
    self, params: Mapping[str, jax.Array]
  ) -> Mapping[str, jax.Array]:
    return params

  def get_config(self) -> Mapping[str, object]:
    return {"name": self.name}


def _count_params(params) -> int:
  return sum(int(x.size) for x in jax.tree.leaves(params))


def _write_synthetic_corpus(
  data_dir: str, sample_rate: int, num_samples: int
) -> None:
  """Writes synthetic AM-modulated harmonic speech-like clips."""
  t = jnp.arange(num_samples, dtype=jnp.float32) / sample_rate
  for split, base_seed in (("train", 10), ("test", 20)):
    clips = []
    for i in range(8):
      f0 = 120.0 + 25.0 * i
      harmonics = sum(
        (1.0 / k) * jnp.sin(2.0 * jnp.pi * (k * f0) * t) for k in range(1, 7)
      )
      env = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * 3.0 * t))
      jitter = 0.01 * jax.random.normal(
        jax.random.PRNGKey(base_seed + i), (num_samples,)
      )
      waveform = 0.05 * env * harmonics + jitter
      clips.append(
        data_clips.Clip(
          audio=waveform,
          sample_rate=sample_rate,
          clip_id=f"synth_{split}_{i}.wav",
        )
      )
    records.write_clips(clips, os.path.join(data_dir, split))


def _load_resampled_wav(path: str, target_sr: int) -> list[np.ndarray]:
  """Loads a WAV file and resamples each channel to `target_sr`."""
  data, orig_sr = sf.read(path, always_2d=True)
  channels = []
  for ch in range(data.shape[1]):
    sig = data[:, ch].astype(np.float64)
    if orig_sr != target_sr:
      g = int(np.gcd(orig_sr, target_sr))
      sig = resample_poly(sig, target_sr // g, orig_sr // g)
    peak = np.max(np.abs(sig)) + 1e-8
    sig = (0.25 * sig / peak).astype(np.float32)
    channels.append(sig)
  return channels


def _write_real_wav_corpus(
  data_dir: str, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
  """Builds disjoint train/test shards from CARFAC's real WAV recordings.

  Returns:
    Tuple of `(clean_excerpt, noisy_excerpt)` (1.024 s at 65 dB SPL, SNR=3 dB)
    from held-out `binaural_test.wav` for perceptual + cochleogram analysis.
  """
  root = "/Users/shaikath/Downloads/viking-main/carfac/test_data"
  long_chs = _load_resampled_wav(
    os.path.join(root, "long_test.wav"), sample_rate
  )
  bin_chs = _load_resampled_wav(
    os.path.join(root, "binaural_test.wav"), sample_rate
  )

  # Split long_test.wav: first 80% -> train segments (1.28 s each), last 20% -> test
  split_idx = int(0.8 * len(long_chs[0]))
  seg_len = int(1.28 * sample_rate)
  train_clips = []
  for ch_idx, ch in enumerate(long_chs):
    train_part = ch[:split_idx]
    for s_idx, start in enumerate(
      range(0, len(train_part) - seg_len + 1, seg_len // 2)
    ):
      seg = train_part[start : start + seg_len]
      if float(np.sqrt(np.mean(seg**2))) > 0.01:
        train_clips.append(
          data_clips.Clip(
            audio=jnp.asarray(seg),
            sample_rate=sample_rate,
            clip_id=f"long_ch{ch_idx}_seg{s_idx}.wav",
          )
        )

  test_clips = []
  for ch_idx, ch in enumerate(long_chs):
    tail = ch[split_idx:]
    if len(tail) >= sample_rate:
      test_clips.append(
        data_clips.Clip(
          audio=jnp.asarray(tail),
          sample_rate=sample_rate,
          clip_id=f"long_heldout_ch{ch_idx}.wav",
        )
      )
  test_clips.append(
    data_clips.Clip(
      audio=jnp.asarray(bin_chs[0]),
      sample_rate=sample_rate,
      clip_id="binaural_heldout.wav",
    )
  )

  records.write_clips(train_clips, os.path.join(data_dir, "train"))
  records.write_clips(test_clips, os.path.join(data_dir, "test"))

  # Extract an active 1.024s (16384-sample, exact SEANet stride multiple) excerpt
  # from held-out binaural_test.wav calibrated to ~65 dB SPL (-42 dBFS RMS)
  excerpt_len = 16384 if sample_rate == 16000 else 32768
  raw = bin_chs[0][:excerpt_len].copy()
  target_rms = 10.0 ** ((65.0 - 107.01) / 20.0)  # ~0.00793 (~65 dB SPL)
  clean_excerpt = (
    raw * (target_rms / (float(np.sqrt(np.mean(raw**2))) + 1e-8))
  ).astype(np.float32)
  rng = np.random.default_rng(123)
  noise = rng.standard_normal(excerpt_len).astype(np.float32)
  snr_db = 3.0
  noise_rms = target_rms * (10.0 ** (-snr_db / 20.0))
  noise_scaled = noise * (
    noise_rms / (float(np.sqrt(np.mean(noise**2))) + 1e-8)
  )
  noisy_excerpt = (clean_excerpt + noise_scaled).astype(np.float32)
  return clean_excerpt, noisy_excerpt


def _build_models(
  sr: int,
) -> tuple[
  shaca_jax.SHACAModel, list[tuple[str, float, float, audio_model.AudioModel]]
]:
  hop_2ms = max(16, round(0.002 * sr))
  shaca_frame_len = 16 if sr <= 16000 else 32
  shaca_model = shaca_jax.SHACAModel(
    sample_rate_hz=float(sr),
    frame_len=shaca_frame_len,
    init_mode="prescription",
    freeze_thresholds=False,
  )
  candidates: list[tuple[str, float, float, audio_model.AudioModel]] = [
    ("SHACA (DAL-Trained)", shaca_model.latency_ms, 5e-2, shaca_model),
    (
      "Viking (Min-Phase)",
      1000.0 * hop_2ms / sr,
      1.2e-3,
      viking_jax.VikingModel(
        module=viking_jax.VikingGenerator(
          use_low_latency_filterbanks=True,
          convert_to_min_phase=True,
          shift_factor=hop_2ms,
          num_subbands=256 if sr <= 16000 else 512,
          prototype_filter_length=256 if sr <= 16000 else 512,
          num_taps=128 if sr <= 16000 else 256,
          lstm_depth=2,
          lstm_units=32,
          hub_units=8,
          max_mask_gain=20.0,
          init_mask_gain=1.8,
        )
      ),
    ),
    (
      "Viking (Linear-Phase)",
      1000.0 * (hop_2ms + (128 if sr <= 16000 else 256) // 2) / sr,
      1.2e-3,
      viking_jax.VikingModel(
        module=viking_jax.VikingGenerator(
          use_low_latency_filterbanks=True,
          convert_to_min_phase=False,
          shift_factor=hop_2ms,
          num_subbands=256 if sr <= 16000 else 512,
          prototype_filter_length=256 if sr <= 16000 else 512,
          num_taps=128 if sr <= 16000 else 256,
          lstm_depth=2,
          lstm_units=32,
          hub_units=8,
          max_mask_gain=20.0,
          init_mask_gain=1.8,
        )
      ),
    ),
    (
      "SEANet (Causal Conv)",
      1000.0 * seanet.total_stride(seanet.SEANetGenerator()) / sr,
      2e-4,
      seanet.SEANetModel(),
    ),
  ]
  return shaca_model, candidates


def _score_perceptual(
  clean_1d: np.ndarray, est_1d: np.ndarray, sr: int
) -> dict[str, float]:
  length = min(len(clean_1d), len(est_1d))
  ctx = metric_context.Context(
    reference=np.asarray(clean_1d[:length], dtype=np.float64),
    estimation=np.asarray(est_1d[:length], dtype=np.float64),
    sample_rate=sr,
    hearing_loss_db=perceptual.HALF_OHC_HEARING_LOSS_DB,
  )
  return {
    "haspi": perceptual.haspi(ctx),
    "hasqi": perceptual.hasqi(ctx),
    "stoi": perceptual.stoi(ctx),
  }


def _run_suite(
  corpus_dir: str,
  sr: int,
  ohc_health: float,
  max_steps: int,
  eval_every: int,
  l1_wav_weight: float,
) -> tuple[
  list[dict], dict[str, tuple[audio_model.AudioModel, object]], dict[str, dict]
]:
  data_cfg = pipeline.DataConfig(
    sources=(processed.ProcessedClips(corpus_dir),),
    transforms=(
      transforms.BlendSource(
        sources=(color_noise.ColorNoise(color="white", sample_rate=sr),),
        snr_db=(0.0, 6.0),
        num_sources=(1, 1),
      ),
      transforms.NormalizeLevel(level_db_spl=(60.0, 72.0)),
      transforms.CosineWindow(ramp_seconds=0.005),
    ),
    global_batch_size=4,
    sample_rate=sr,
    chunk_seconds=0.128,
    max_epochs=80,
    seed=42,
  )

  cochleae = cochlea.CochleaPair(
    healthy=carfac.build(sr),
    impaired=carfac.build(sr, ohc_health=ohc_health),
  )
  ctx = auditory_context.build(cochleae)

  train_terms: tuple[terms_lib.LossTerm, ...] = (
    representation_losses.SaiSsimLoss(weight=1.0, timewinsz=36, chanwinsz=6),
    representation_losses.SaiPndLoss(weight=1.0, alpha=0.5, beta=0.8),
    representation_losses.NapL1Loss(weight=1.0),
    waveform_losses.L1WaveformLoss(weight=l1_wav_weight),
    waveform_losses.LowFrequencyDriftLoss(weight=2.0, sample_rate=sr),
  )
  eval_terms = train_terms

  val_batches = list(data_lib.validation_batches(data_cfg, num_val_examples=8))

  baseline = _IdentityAudioModel()
  base_eval_fn = steps.build_eval_step(baseline.apply, eval_terms, ctx)
  base_m = loop.run_validation_epoch({}, val_batches, base_eval_fn, eval_terms)

  shaca_model, candidates = _build_models(sr)
  shaca_init_params = shaca_model.init(jax.random.PRNGKey(0))
  shaca_eval_fn = steps.build_eval_step(shaca_model.apply, eval_terms, ctx)
  shaca_presc_m = loop.run_validation_epoch(
    shaca_init_params, val_batches, shaca_eval_fn, eval_terms
  )

  rows: list[dict] = [
    {
      "label": "Unprocessed (Impaired)",
      "params": 0,
      "latency_ms": 0.0,
      "ssim": float(base_m["val_ssim_loss"]),
      "pnd": float(base_m["val_pnd_sai_loss"]),
      "nap_l1": float(base_m["val_l1_nap_loss"]),
      "aud_total": float(
        base_m["val_ssim_loss"]
        + base_m["val_pnd_sai_loss"]
        + base_m["val_l1_nap_loss"]
      ),
    },
    {
      "label": "SHACA (Prescribed)",
      "params": _count_params(shaca_init_params),
      "latency_ms": shaca_model.latency_ms,
      "ssim": float(shaca_presc_m["val_ssim_loss"]),
      "pnd": float(shaca_presc_m["val_pnd_sai_loss"]),
      "nap_l1": float(shaca_presc_m["val_l1_nap_loss"]),
      "aud_total": float(
        shaca_presc_m["val_ssim_loss"]
        + shaca_presc_m["val_pnd_sai_loss"]
        + shaca_presc_m["val_l1_nap_loss"]
      ),
    },
  ]
  trained_artifacts: dict[str, tuple[audio_model.AudioModel, object]] = {
    "Unprocessed (Impaired)": (baseline, {}),
    "SHACA (Prescribed)": (shaca_model, shaca_init_params),
  }
  curves: dict[str, dict] = {}

  for label, latency_ms, lr, model in candidates:
    optimizer = optax.adam(lr, b1=0.5, b2=0.9)
    res = loop.fit(
      model=model,
      data=pipeline.build(data_cfg),
      loss=train_terms,
      context=ctx,
      optimizer=optimizer,
      max_steps=max_steps,
      eval_every=eval_every,
      val_data=val_batches,
      seed=0,
    )
    eval_fn = steps.build_eval_step(model.apply, eval_terms, ctx)
    m = loop.run_validation_epoch(res.params, val_batches, eval_fn, eval_terms)
    rows.append(
      {
        "label": label,
        "params": _count_params(res.params),
        "latency_ms": latency_ms,
        "ssim": float(m["val_ssim_loss"]),
        "pnd": float(m["val_pnd_sai_loss"]),
        "nap_l1": float(m["val_l1_nap_loss"]),
        "aud_total": float(
          m["val_ssim_loss"] + m["val_pnd_sai_loss"] + m["val_l1_nap_loss"]
        ),
      }
    )
    trained_artifacts[label] = (model, res.params)
    aud_series = [
      float(s + p + n)
      for s, p, n in zip(
        res.history["val_ssim_loss"],
        res.history["val_pnd_sai_loss"],
        res.history["val_l1_nap_loss"],
      )
    ]
    curves[label] = {
      "steps": list(res.val_steps),
      "aud_total": aud_series,
      "ssim": [float(v) for v in res.history["val_ssim_loss"]],
      "pnd": [float(v) for v in res.history["val_pnd_sai_loss"]],
      "nap_l1": [float(v) for v in res.history["val_l1_nap_loss"]],
    }

  return rows, trained_artifacts, curves


def _plot_fig1_overview(
  rows: list[dict], sr: int, ohc: float, steps_n: int, out_path: str
) -> None:
  """Figure 1: Architecture (Latency vs Params) + CARFAC Metrics + Scorecard."""
  plt.rcParams.update(
    {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10}
  )
  fig = plt.figure(figsize=(15.5, 9.2), dpi=300)
  gs = fig.add_gridspec(
    2, 2, height_ratios=[1.25, 0.85], hspace=0.38, wspace=0.25
  )

  # Panel A: Latency vs Parameter Count Pareto Frontier
  ax_a = fig.add_subplot(gs[0, 0])
  ax_a.axhspan(
    0.0,
    5.0,
    color="#E8F8F5",
    alpha=0.65,
    zorder=0,
    label="Hearable Safe Zone (< 5 ms)",
  )
  ax_a.axhline(5.0, color="#C0392B", linestyle="--", linewidth=1.3, zorder=1)
  ax_a.text(
    70,
    5.45,
    "5 ms Hearing-Aid Comb-Filtering Threshold (Stone & Moore, 1999)",
    color="#C0392B",
    fontsize=8.5,
    fontweight="bold",
  )

  offsets = {
    "SHACA (Prescribed)": (12, 8),
    "SHACA (DAL-Trained)": (12, -14),
    "Viking (Min-Phase)": (10, 8),
    "Viking (Linear-Phase)": (-55, 10),
    "SEANet (Causal Conv)": (-118, -16),
  }
  for r in rows:
    if r["params"] == 0:
      continue
    lbl = r["label"]
    marker = "D" if "Prescribed" in lbl else "o"
    ax_a.scatter(
      r["params"],
      r["latency_ms"],
      s=130,
      color=_COLORS[lbl],
      marker=marker,
      edgecolors="black",
      linewidths=0.9,
      zorder=4,
    )
    dx, dy = offsets.get(lbl, (8, 6))
    ax_a.annotate(
      f"{lbl}\n({r['params']:,} params, {r['latency_ms']:.2f} ms)",
      (r["params"], r["latency_ms"]),
      xytext=(dx, dy),
      textcoords="offset points",
      fontsize=8.2,
      fontweight="semibold",
      bbox=dict(
        boxstyle="round,pad=0.2", fc="white", ec=_COLORS[lbl], alpha=0.88
      ),
    )

  ax_a.set_xscale("log")
  ax_a.set_xlim(25, 4e7)
  ax_a.set_ylim(-1.0, 18.5)
  ax_a.set_xlabel("Trainable Parameter Count (log scale)")
  ax_a.set_ylabel("Algorithmic Frame/Cascade Latency (ms)")
  ax_a.set_title(
    "(a) Hearing-Aid Architecture Footprint: Latency vs. Parameters",
    fontweight="bold",
  )
  ax_a.grid(True, which="both", linestyle=":", alpha=0.55)
  ax_a.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

  # Panel B: Grouped CARFAC Auditory Losses
  ax_b = fig.add_subplot(gs[0, 1])
  short_labels = [
    "Unprocessed\n(Impaired)",
    "SHACA\n(Prescribed)",
    "SHACA\n(DAL)",
    "Viking\n(Min-Phase)",
    "Viking\n(Lin-Phase)",
    "SEANet\n(Causal)",
  ]
  x = np.arange(len(rows))
  width = 0.25
  ssim_vals = [r["ssim"] for r in rows]
  pnd_vals = [r["pnd"] for r in rows]
  nap_vals = [r["nap_l1"] for r in rows]

  b1 = ax_b.bar(
    x - width,
    ssim_vals,
    width,
    label="SAI SSIM Loss (↓)",
    color="#2E86C1",
    edgecolor="black",
    linewidth=0.6,
  )
  b2 = ax_b.bar(
    x,
    pnd_vals,
    width,
    label="SAI PND Loss (↓)",
    color="#E67E22",
    edgecolor="black",
    linewidth=0.6,
  )
  b3 = ax_b.bar(
    x + width,
    nap_vals,
    width,
    label="NAP L1 Loss (↓)",
    color="#27AE60",
    edgecolor="black",
    linewidth=0.6,
  )

  for bars in (b1, b2, b3):
    for bar in bars:
      h = bar.get_height()
      ax_b.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + 0.012,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=7.2,
      )

  ax_b.set_xticks(x)
  ax_b.set_xticklabels(short_labels, fontsize=8.5)
  ax_b.set_ylim(0.0, max(*ssim_vals, *pnd_vals, *nap_vals) * 1.18)
  ax_b.set_ylabel(
    "CARFAC Cochlear Representation Loss (Lower = Closer to Healthy)"
  )
  ax_b.set_title(
    f"(b) CARFAC Auditory Restoration Losses ({sr // 1000} kHz, OHC={ohc}, {steps_n} DAL Steps)",
    fontweight="bold",
  )
  ax_b.grid(axis="y", linestyle=":", alpha=0.6)
  ax_b.legend(loc="upper right", fontsize=8.5, ncol=3, framealpha=0.92)

  # Panel C: Quantitative Summary Scorecard Table
  ax_c = fig.add_subplot(gs[1, :])
  ax_c.axis("off")
  col_labels = [
    "Model / Paradigm",
    "Params",
    "Latency",
    "SAI SSIM (↓)",
    "SAI PND (↓)",
    "NAP L1 (↓)",
    "Auditory Sum (↓)",
    "HASPI v2 (↑)",
    "HASQI v2 (↑)",
    "STOI (↑)",
  ]
  base_aud = rows[0]["aud_total"]
  cell_text = []
  for r in rows:
    delta_pct = 100.0 * (r["aud_total"] - base_aud) / base_aud
    delta_str = "baseline" if r["params"] == 0 else f"{delta_pct:+.1f}%"
    cell_text.append(
      [
        r["label"],
        f"{r['params']:,}",
        f"{r['latency_ms']:.2f} ms",
        f"{r['ssim']:.4f}",
        f"{r['pnd']:.4f}",
        f"{r['nap_l1']:.4f}",
        f"{r['aud_total']:.4f} ({delta_str})",
        f"{r.get('haspi', float('nan')):.3f}",
        f"{r.get('hasqi', float('nan')):.3f}",
        f"{r.get('stoi', float('nan')):.3f}",
      ]
    )

  tbl = ax_c.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.02, 1.0, 0.84],
  )
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(8.8)
  col_widths = [0.18, 0.08, 0.08, 0.09, 0.09, 0.09, 0.15, 0.08, 0.08, 0.08]
  for (row_idx, col_idx), cell in tbl.get_celld().items():
    cell.set_width(col_widths[col_idx])
    cell.set_edgecolor("#BDC3C7")
    if row_idx == 0:
      cell.set_facecolor("#2C3E50")
      cell.get_text().set_color("white")
      cell.get_text().set_fontweight("bold")
    else:
      lbl = rows[row_idx - 1]["label"]
      cell.set_facecolor("#F8F9F9" if row_idx % 2 == 1 else "#FFFFFF")
      if col_idx == 0:
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_color(_COLORS[lbl])

  ax_c.set_title(
    "(c) End-to-End Quantitative Summary Across Classical WDRC (SHACA),"
    " Ultra-Low-Latency Neural (Viking), and U-Net (SEANet)",
    fontweight="bold",
    pad=6,
  )
  fig.suptitle(
    "Differentiable Auditory Loop (DAL) Model Comparison: Architecture,"
    " Cochlear Losses & Perceptual Scores",
    fontsize=13.5,
    fontweight="bold",
    y=0.98,
  )
  fig.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def _plot_fig2_nap_heatmaps(
  clean_1d: np.ndarray,
  noisy_1d: np.ndarray,
  trained_artifacts: dict[str, tuple[audio_model.AudioModel, object]],
  sr: int,
  ohc: float,
  out_path: str,
) -> None:
  """Figure 2: Side-by-side CARFAC NAP Cochleogram Heatmaps on Real Audio."""
  healthy_carfac = carfac.build(sr)
  impaired_carfac = carfac.build(sr, ohc_health=ohc)
  pole_freqs = np.asarray(healthy_carfac.hypers.ears[0].pole_freqs)[::-1]

  def get_nap(
    sig_1d: np.ndarray, cf_model: carfac.CarfacCochleaModel
  ) -> np.ndarray:
    naps = np.asarray(jnp.squeeze(cf_model(jnp.asarray(sig_1d)), axis=-1))
    return naps[:, ::-1]  # shape (T, C) ordered Low CF -> High CF

  # Zoom to an expressive 0.60 s speech/modulation window so fine structure is crisp
  win_samples = int(0.60 * sr)
  clean_seg = clean_1d[:win_samples]
  noisy_seg = noisy_1d[:win_samples]
  healthy_nap = get_nap(clean_seg, healthy_carfac)
  impaired_unproc_nap = get_nap(noisy_seg, impaired_carfac)

  panel_specs = [
    (
      "1. Clean Target → Healthy CARFAC (OHC = 1.0)",
      clean_seg,
      healthy_nap,
      True,
    ),
    (
      "2. Noisy Input → Impaired CARFAC (OHC = 0.5, Unprocessed)",
      noisy_seg,
      impaired_unproc_nap,
      False,
    ),
  ]
  for key, title in [
    ("SHACA (Prescribed)", "3. SHACA (Prescribed, 0.19 ms) → Impaired CARFAC"),
    (
      "SHACA (DAL-Trained)",
      "4. SHACA (DAL-Trained, 0.19 ms) → Impaired CARFAC",
    ),
    ("Viking (Min-Phase)", "5. Viking (Min-Phase, 2.00 ms) → Impaired CARFAC"),
    (
      "SEANet (Causal Conv)",
      "6. SEANet (Causal Conv, 16.00 ms) → Impaired CARFAC",
    ),
  ]:
    model, params = trained_artifacts[key]
    aided_full = np.asarray(
      jnp.squeeze(model.apply(params, jnp.asarray(noisy_1d[None, :, None])))
    )
    aided_seg = aided_full[:win_samples]
    aided_nap = get_nap(aided_seg, impaired_carfac)
    panel_specs.append((title, aided_seg, aided_nap, False))

  vmax = float(np.percentile(healthy_nap, 99.2))
  fig = plt.figure(figsize=(15.5, 10.5), dpi=300)
  gs = fig.add_gridspec(
    4,
    2,
    height_ratios=[0.48, 1.0, 1.0, 1.0],
    hspace=0.38,
    wspace=0.16,
  )

  # Top strip: Clean vs Noisy vs SHACA/Viking waveforms
  ax_wav = fig.add_subplot(gs[0, :])
  t_ms = 1000.0 * np.arange(win_samples) / sr
  ax_wav.plot(
    t_ms,
    noisy_seg,
    color="#BDC3C7",
    linewidth=0.7,
    label="Noisy Input (SNR = 3 dB, 65 dB SPL)",
  )
  ax_wav.plot(
    t_ms,
    clean_seg,
    color="#2C3E50",
    linewidth=0.85,
    label="Clean Reference (binaural_test.wav)",
  )
  ax_wav.set_xlim(0.0, t_ms[-1])
  ax_wav.set_ylabel("Amplitude")
  ax_wav.set_xlabel("Time (ms)")
  ax_wav.set_title(
    "Input Acoustic Stimulus (CARFAC Real Test Recording + White Noise at 3 dB SNR)",
    fontweight="bold",
  )
  ax_wav.legend(loc="upper right", ncol=2, fontsize=8.5, framealpha=0.9)
  ax_wav.grid(True, linestyle=":", alpha=0.5)

  # Frequency tick positions along flipped CARFAC channel axis (Low -> High Hz)
  target_hz = [50, 150, 400, 1000, 2500, 6000]
  ytick_idx = [int(np.argmin(np.abs(pole_freqs - f))) for f in target_hz]
  ytick_lbl = [f"{int(round(pole_freqs[i]))} Hz" for i in ytick_idx]

  im = None
  for idx, (title, _, nap_mat, is_ref) in enumerate(panel_specs):
    r_i = 1 + (idx // 2)
    c_i = idx % 2
    ax = fig.add_subplot(gs[r_i, c_i])
    im = ax.imshow(
      nap_mat.T,
      aspect="auto",
      origin="lower",
      cmap="magma",
      vmin=0.0,
      vmax=vmax,
      extent=[0.0, t_ms[-1], 0, nap_mat.shape[1] - 1],
    )
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(ytick_lbl, fontsize=8)
    if c_i == 0:
      ax.set_ylabel("Cochlear CF")
    if r_i == 3:
      ax.set_xlabel("Time (ms)")

    if is_ref:
      badge = "Reference Target"
    else:
      nap_l1 = float(np.mean(np.abs(nap_mat - healthy_nap)))
      firing_ratio = (
        100.0 * float(np.mean(nap_mat)) / (float(np.mean(healthy_nap)) + 1e-8)
      )
      badge = f"NAP L1 = {nap_l1:.3f} | Mean Neural Drive = {firing_ratio:.0f}% of Healthy"
    ax.set_title(f"{title}\n[{badge}]", fontsize=9.3, fontweight="bold")

  cbar_ax = fig.add_axes([0.92, 0.08, 0.014, 0.68])
  assert im is not None
  cbar = fig.colorbar(im, cax=cbar_ax)
  cbar.set_label(
    "CARFAC Neural Activity Pattern (NAP) Firing Rate", fontsize=9.5
  )

  fig.suptitle(
    "Cochlear Neural Activity Pattern (NAP) Restoration on Real Audio (16 kHz, Impaired OHC = 0.5)",
    fontsize=13.5,
    fontweight="bold",
    y=0.98,
  )
  fig.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def _plot_fig3_smoke_vs_real(
  smoke_rows: list[dict],
  real_rows: list[dict],
  real_curves: dict[str, dict],
  smoke_steps: int,
  real_steps: int,
  out_path: str,
) -> None:
  """Figure 3: Quick Synthetic Smoke Run vs. Real-WAV Multi-Step Training."""
  fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=300)
  fig.subplots_adjust(wspace=0.28)

  # Panel A: Real-WAV Convergence Curves
  ax0 = axes[0]
  unproc_val = real_rows[0]["aud_total"]
  presc_val = real_rows[1]["aud_total"]
  ax0.axhline(
    unproc_val,
    color=_COLORS["Unprocessed (Impaired)"],
    linestyle="--",
    linewidth=1.4,
    label=f"Unprocessed ({unproc_val:.2f})",
  )
  ax0.axhline(
    presc_val,
    color=_COLORS["SHACA (Prescribed)"],
    linestyle=":",
    linewidth=1.6,
    label=f"SHACA Prescribed ({presc_val:.2f})",
  )
  for lbl, series in real_curves.items():
    ax0.plot(
      series["steps"],
      series["aud_total"],
      marker="o",
      markersize=5,
      linewidth=2.0,
      color=_COLORS[lbl],
      label=lbl,
    )
  ax0.set_xlabel("Adam Optimization Step (Real CARFAC WAV Corpus)")
  ax0.set_ylabel("Validation Cochlear Loss (SSIM + PND + NAP L1 ↓)")
  ax0.set_title(
    f"(a) Real-WAV DAL Training Convergence (0 → {real_steps} Steps)",
    fontweight="bold",
  )
  ax0.grid(True, linestyle=":", alpha=0.6)
  ax0.legend(loc="best", fontsize=7.8, framealpha=0.92)

  # Panel B: Synthetic Smoke (5 steps) vs Real-WAV (30 steps) % Improvement
  ax1 = axes[1]
  compare_labels = [
    "SHACA (Prescribed)",
    "SHACA (DAL-Trained)",
    "Viking (Min-Phase)",
    "Viking (Linear-Phase)",
    "SEANet (Causal Conv)",
  ]
  short_cmp = [
    "SHACA\n(Presc.)",
    "SHACA\n(DAL)",
    "Viking\n(Min-Ph)",
    "Viking\n(Lin-Ph)",
    "SEANet\n(Causal)",
  ]
  smoke_map = {r["label"]: r for r in smoke_rows}
  real_map = {r["label"]: r for r in real_rows}
  smoke_base = smoke_map["Unprocessed (Impaired)"]["aud_total"]
  real_base = real_map["Unprocessed (Impaired)"]["aud_total"]

  # Positive % = reduction in cochlear loss relative to Unprocessed (better!)
  smoke_imp = [
    100.0 * (smoke_base - smoke_map[k]["aud_total"]) / smoke_base
    for k in compare_labels
  ]
  real_imp = [
    100.0 * (real_base - real_map[k]["aud_total"]) / real_base
    for k in compare_labels
  ]

  x = np.arange(len(compare_labels))
  w = 0.35
  b_s = ax1.bar(
    x - w / 2,
    smoke_imp,
    w,
    label=f"Synthetic Smoke ({smoke_steps} steps)",
    color="#AED6F1",
    edgecolor="black",
    linewidth=0.7,
  )
  b_r = ax1.bar(
    x + w / 2,
    real_imp,
    w,
    label=f"Real WAV Corpus ({real_steps} steps)",
    color="#1F618D",
    edgecolor="black",
    linewidth=0.7,
  )
  ax1.axhline(0.0, color="black", linewidth=0.9)
  for bars in (b_s, b_r):
    for bar in bars:
      v = bar.get_height()
      va = "bottom" if v >= 0 else "top"
      off = 0.8 if v >= 0 else -1.2
      ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        v + off,
        f"{v:+.1f}%",
        ha="center",
        va=va,
        fontsize=7.3,
        fontweight="semibold",
      )
  ax1.set_xticks(x)
  ax1.set_xticklabels(short_cmp, fontsize=8.3)
  ymin = min(*smoke_imp, *real_imp, 0.0)
  ymax = max(*smoke_imp, *real_imp, 5.0)
  ax1.set_ylim(ymin - 8.0, ymax + 8.0)
  ax1.set_ylabel("Cochlear Loss Reduction vs. Unprocessed (%, Higher = Better)")
  ax1.set_title(
    "(b) Synthetic Smoke vs. Real-WAV Relative Gain", fontweight="bold"
  )
  ax1.grid(axis="y", linestyle=":", alpha=0.6)
  ax1.legend(loc="best", fontsize=8.2, framealpha=0.92)

  # Panel C: Held-Out Real-WAV Perceptual Metrics (HASPI v2, HASQI v2, STOI)
  ax2 = axes[2]
  all_short = [
    "Unproc.",
    "SHACA\n(Presc.)",
    "SHACA\n(DAL)",
    "Viking\n(Min-Ph)",
    "Viking\n(Lin-Ph)",
    "SEANet",
  ]
  x_all = np.arange(len(real_rows))
  w3 = 0.25
  haspi_v = [r["haspi"] for r in real_rows]
  hasqi_v = [r["hasqi"] for r in real_rows]
  stoi_v = [r["stoi"] for r in real_rows]

  ax2.bar(
    x_all - w3,
    haspi_v,
    w3,
    label="HASPI v2 (Intelligibility ↑)",
    color="#8E44AD",
    edgecolor="black",
    linewidth=0.6,
  )
  ax2.bar(
    x_all,
    hasqi_v,
    w3,
    label="HASQI v2 (Quality ↑)",
    color="#16A085",
    edgecolor="black",
    linewidth=0.6,
  )
  ax2.bar(
    x_all + w3,
    stoi_v,
    w3,
    label="STOI (↑)",
    color="#F39C12",
    edgecolor="black",
    linewidth=0.6,
  )
  ax2.set_xticks(x_all)
  ax2.set_xticklabels(all_short, fontsize=8.2)
  ax2.set_ylim(0.0, 1.08)
  ax2.set_ylabel("Perceptual Score on [0, 1] (Higher = Better)")
  ax2.set_title(
    "(c) Held-Out Real-WAV Perceptual Scores (OHC = 0.5)", fontweight="bold"
  )
  ax2.grid(axis="y", linestyle=":", alpha=0.6)
  ax2.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

  fig.suptitle(
    "Synthetic Smoke Run vs. Real-WAV Multi-Step DAL Training & Audiological Perceptual Evaluation",
    fontsize=13.2,
    fontweight="bold",
    y=1.02,
  )
  fig.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.close(fig)


def main(argv: Sequence[str]) -> None:
  del argv
  sr = _SAMPLE_RATE.value
  ohc = _OHC_HEALTH.value
  out_dir = _OUTPUT_DIR.value
  os.makedirs(out_dir, exist_ok=True)

  with tempfile.TemporaryDirectory() as tmpdir:
    synth_dir = os.path.join(tmpdir, "synth")
    real_dir = os.path.join(tmpdir, "real")
    _write_synthetic_corpus(synth_dir, sr, round(0.256 * sr))
    clean_excerpt, noisy_excerpt = _write_real_wav_corpus(real_dir, sr)

    print(
      f"--- 1/3 Running Synthetic Smoke Suite ({_SMOKE_STEPS.value} steps) ---"
    )
    smoke_rows, _, _ = _run_suite(
      synth_dir,
      sr=sr,
      ohc_health=ohc,
      max_steps=_SMOKE_STEPS.value,
      eval_every=_SMOKE_STEPS.value,
      l1_wav_weight=0.0,
    )

    print(
      f"--- 2/3 Running Real-WAV Training Suite ({_REAL_STEPS.value} steps) ---"
    )
    real_rows, real_artifacts, real_curves = _run_suite(
      real_dir,
      sr=sr,
      ohc_health=ohc,
      max_steps=_REAL_STEPS.value,
      eval_every=_EVAL_EVERY.value,
      l1_wav_weight=0.0,
    )

    print(
      "--- 3/3 Scoring HASPI v2, HASQI v2, and STOI on Held-Out Real Audio ---"
    )
    for r in real_rows:
      model, params = real_artifacts[r["label"]]
      est = np.asarray(
        jnp.squeeze(
          model.apply(params, jnp.asarray(noisy_excerpt[None, :, None]))
        )
      )
      scores = _score_perceptual(clean_excerpt, est, sr)
      r.update(scores)

    fig1_path = os.path.join(
      out_dir, "fig1_architecture_and_metrics_overview.png"
    )
    fig2_path = os.path.join(out_dir, "fig2_cochlear_nap_heatmaps.png")
    fig3_path = os.path.join(out_dir, "fig3_smoke_vs_real_wav_training.png")
    summary_json = os.path.join(out_dir, "benchmark_summary.json")

    _plot_fig1_overview(real_rows, sr, ohc, _REAL_STEPS.value, fig1_path)
    _plot_fig2_nap_heatmaps(
      clean_excerpt, noisy_excerpt, real_artifacts, sr, ohc, fig2_path
    )
    _plot_fig3_smoke_vs_real(
      smoke_rows,
      real_rows,
      real_curves,
      _SMOKE_STEPS.value,
      _REAL_STEPS.value,
      fig3_path,
    )

    with open(summary_json, "w") as f:
      json.dump(
        {
          "sample_rate_hz": sr,
          "impaired_ohc_health": ohc,
          "smoke_steps": _SMOKE_STEPS.value,
          "real_steps": _REAL_STEPS.value,
          "smoke_rows": smoke_rows,
          "real_rows": real_rows,
          "real_curves": real_curves,
        },
        f,
        indent=2,
      )

    print(
      f"Saved:\n  {fig1_path}\n  {fig2_path}\n  {fig3_path}\n  {summary_json}"
    )


if __name__ == "__main__":
  app.run(main)
