# Sony SampleID proof of concept

This folder sets up Sony's open-source **SampleID** model for the Lastrada Applied AI Lab project.

Sony repository: https://github.com/sony/sampleid  
Pretrained checkpoint: https://zenodo.org/records/17413869

## What this POC does

Given a **known original song** and a **known sampled/derivative song**, the script:

1. converts both songs to mono 16 kHz audio;
2. slices them into overlapping 5-second windows;
3. runs every window through Sony's released pretrained SampleID model;
4. compares the resulting embeddings; and
5. prints the strongest candidate source ↔ derivative timestamp pairs.

This is intended as **Step 1 candidate localization**, not as a final Nature of Use determination or legal-grade boundary detector. A later refinement step should tighten start/end boundaries and compute total occupied duration / coverage.

## Setup

Sony currently declares Python **3.12+**.

```bash
cd sony_sampleid
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first run will download Sony's published pretrained checkpoint automatically.

## Run

```bash
python compare_songs.py \
  /path/to/original.wav \
  /path/to/derivative.wav \
  --top-k 20 \
  --csv-out results/matches.csv
```

Defaults:
- chunk size: **5.0 seconds**
- hop size: **2.5 seconds**
- output: top **20** matching window pairs

You can change the granularity:

```bash
python compare_songs.py original.wav derivative.wav \
  --chunk-seconds 5 \
  --hop-seconds 1
```

A smaller hop gives denser candidate timestamps but increases compute and does **not** by itself make the boundaries more accurate.

## Expected output

```text
similarity | original (s) | derivative (s)
----------------------------------------------------------
0.91       | 42.50-47.50  | 70.00-75.00
0.89       | 42.50-47.50  | 95.00-100.00
...
```

These are candidate matching windows to pass into the next timing-refinement / Nature of Use step.

## Important limitations

- Sony's released inference model compares chunks; it does not directly return exact sample boundaries.
- Similarity scores need to be calibrated on Lastrada-approved song pairs before using any fixed threshold.
- This POC does not calculate clearance conclusions, prominence, instrument identity, or legal/business split recommendations.
- Do not commit client audio to this repository.
