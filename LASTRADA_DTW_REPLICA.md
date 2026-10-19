# Lastrada Sony + DTW replica

This branch deploys Sri's Lastrada Nature of Use workflow with Sony SampleID and guarded DTW integrated into the normal master-analysis path.

## Streamlit Community Cloud settings

- Repository: `manzaa-glitch/projects`
- Branch: `lastrada-dtw-replica`
- Main file path: `streamlit_app.py`
- Python: `3.12`

## Default analysis flow

For a master-rights analysis, the app now attempts this sequence by default:

1. **Sony SampleID embeddings** compare overlapping windows from the original and new track.
2. **Rank + temporal consistency** keeps only coherent source ↔ derivative sequences; raw cosine values are not treated as probabilities.
3. **Direct waveform alignment** refines nearly unchanged recording reuse.
4. **Guarded DTW** refines candidates that may include moderate pitch or timing changes.
5. If DTW cannot produce a plausible duration/overlap, the app preserves Sony's coarse candidate instead of forcing an alignment.
6. The resulting paired timestamps appear in the existing **Step 3 · Listen & check** review table and continue through the Nature of Use workflow.

PF-NMF remains available as contextual/fallback evidence when Sony cannot run. It is not substituted for paired source timestamps.

## Checkpoint resolution

The app searches for `sampleid-best.ckpt` in this order:

1. the optional path entered under **Demo and advanced options**;
2. `LASTRADA_SAMPLEID_CHECKPOINT`;
3. `checkpoints/sampleid-best.ckpt`;
4. `~/.lastrada_sampleid/sampleid-best.ckpt`;
5. a URL supplied through `LASTRADA_SAMPLEID_CHECKPOINT_URL`, downloaded atomically into the cache.

For a local Windows test, the existing checkpoint is commonly located at:

```text
C:\Users\<username>\.lastrada_sampleid\sampleid-best.ckpt
```

For Streamlit Community Cloud, configure a stable checkpoint mirror as the secret/environment value `LASTRADA_SAMPLEID_CHECKPOINT_URL`. Without a valid checkpoint, the app clearly falls back to PF-NMF instead of pretending Sony or DTW ran.

## Interpretation

- Sony candidate similarity is uncalibrated retrieval evidence, not probability.
- DTW acceptance means only that the proposed alignment passed engineering guardrails.
- All automated intervals remain candidates until a reviewer listens and accepts or edits them.
- Demucs stem separation is not part of this default cloud path.
