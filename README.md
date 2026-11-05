# viking-jax

JAX/Flax implementation of the **Viking** low-latency speech enhancement and
hearing-aid model, based on
[*Towards Sub-millisecond Latency Real-Time Speech Enhancement Models on Hearables*](https://arxiv.org/abs/2409.18239)
(Dementyev et al., ICASSP 2025).

`viking-jax` provides both full-sequence vectorized execution (`__call__`) for
training and chunk-by-chunk stateful execution (`init_stream_state`,
`stream_step`) for real-time streaming inference, with numerical parity across
both modes.

## Features and Hearing-Aid Extensions

In addition to the architecture described in
[arXiv:2409.18239](https://arxiv.org/abs/2409.18239), this implementation
includes extensions for hearing-aid speech enhancement and amplification:

- **Dual Analysis/Synthesis Filterbanks**:
  - **Standard Causal STFT Mode (`use_low_latency_filterbanks=False`)**:
    Causally padded Hann-windowed STFT (`frame_length`, `frame_step`),
    power-law magnitude compression (`in_magnitude_compression`), subband mask
    application, and overlap-add inverse STFT synthesis.
  - **Ultra-Low-Latency Filterbank Mode (`use_low_latency_filterbanks=True`)**:
    Hamming-windowed sinc prototype analysis filterbank
    (`prototype_filter_length`, `shift_factor`, `num_subbands`) paired with
    time-domain FIR synthesis filtering (`num_taps`) and per-frame linear
    crossfading over `shift_factor` samples.
  - **Minimum-Phase FIR Conversion (`convert_to_min_phase=True`)**: Converts
    per-frame linear-phase synthesis FIR taps to minimum phase via the real
    cepstrum and causal Hilbert-transform window, concentrating filter energy
    near tap 0 to achieve sub-millisecond algorithmic latency.
- **Hearing-Aid Amplification Masks (`max_mask_gain`)**:
  - Whereas standard denoising models restrict subband masks to `(0, 1)`,
    `VikingCell` and `VikingGenerator` parameterize the output mask as
    `max_mask_gain * sigmoid(logits)` (default `max_mask_gain = 20.0`).
  - This allows the model to apply frequency-dependent insertion gain
    (amplification above `1.0`) alongside noise suppression.
  - When `max_mask_gain > 1.0`, the mask projection bias is initialized to
    `-log(max_mask_gain - 1.0)` so that an unperturbed network starts at unity
    gain (`1.0`) across all subbands. Setting `max_mask_gain = 1.0` recovers a
    standard `(0, 1)` suppression mask.
- **Multi-Layer Stacked Parallel RNN with Inter-Layer Hubs (`lstm_depth`)**:
  - Splits the frequency bins across `num_parallel` independent LSTM branches
    per layer (`lstm_units` per branch) to keep parameter count and MACs small.
  - Supports stacking multiple recurrent layers (`lstm_depth >= 1`). Between
    stacked layers (`layer_idx < lstm_depth - 1`), branch outputs are
    concatenated and projected through a linear dense hub (`hub_units`) that is
    fed into every branch of the next layer for cross-band context exchange,
    followed by a final `tanh` hub projection after the last layer.
  - Clips LSTM cell states (`c`) to `[-10.0, 10.0]` at each step for numerical
    stability during long-horizon streaming.
- **Configurable Synthesis Tap Count (`num_taps`)**:
  - Controls the windowed synthesis FIR filter length (up to
    `prototype_filter_length // 2`) in low-latency filterbank mode.

## Installation

```bash
pip install -e .
```

Or with [`uv`](https://docs.astral.sh/uv/) for development and testing:

```bash
uv sync --all-groups
uv run pytest
```

## Quickstart

### Full-Sequence Batch Inference

```python
import jax
import jax.numpy as jnp
from viking_jax import VikingGenerator

model = VikingGenerator(
    use_low_latency_filterbanks=True,
    shift_factor=64,
    num_subbands=512,
    prototype_filter_length=512,
    convert_to_min_phase=True,
    num_taps=256,
    num_parallel=4,
    lstm_depth=2,
    lstm_units=32,
    hub_units=8,
    max_mask_gain=20.0,
)

x = jnp.zeros((1, 16000, 1))
variables = model.init(jax.random.PRNGKey(0), x, train=False)
y = model.apply(variables, x, train=False)  # shape: (1, 16000, 1)
```

### Real-Time Chunk-by-Chunk Streaming

`init_stream_state` and `stream_step` process one hop (`shift_factor` samples in
low-latency filterbank mode, or `frame_step` samples in STFT mode) per call and
produce outputs matching full-sequence batch mode:

```python
state = model.init_stream_state(batch_size=1)

@jax.jit
def step_fn(stream_state, audio_chunk):
  return model.apply(
      variables,
      stream_state,
      audio_chunk,
      method=model.stream_step,
  )

chunk = jnp.zeros((1, model.shift_factor, 1))
state, out_chunk = step_fn(state, chunk)  # out_chunk shape: (1, 64, 1)
```

## Authors and Acknowledgments

This JAX/Flax implementation and its hearing-aid extensions were developed by
Shaikat Hossain, Tomo Akitomi, and Oscar Wahltinez.

We thank Artem Dementyev, Chandan K. A. Reddy, Scott Wisdom, Navin Chatlani,
John R. Hershey, and Richard F. Lyon for the original Viking architecture, as
well as Jeremy Thorpe for code review and feedback.

## Reference

If you use this model in your research, please cite the Viking paper:

```
@inproceedings{dementyev2025towards,
  author={Dementyev, Artem and Reddy, Chandan K. A. and Wisdom, Scott and Chatlani, Navin and Hershey, John R. and Lyon, Richard F.},
  title={Towards Sub-millisecond Latency Real-Time Speech Enhancement Models on Hearables},
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  note={arXiv:2409.18239}
}
```

## License and Disclaimers

Licensed under the [Apache License, Version 2.0](LICENSE).

This is not an officially supported Google product.

Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).
