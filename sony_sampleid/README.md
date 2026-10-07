# Sony SampleID proof of concept

This folder sets up Sony's open-source **SampleID** model for the Lastrada Applied AI Lab project.

Sony repository: https://github.com/sony/sampleid  
Pretrained checkpoint: https://zenodo.org/records/17413869

## What this POC does

Given a **known original song** and a **known sampled/derivative song**, the tool:
1. converts both songs to mono 16 kHz audio;
2. slices them into overlapping windows;
3. runs every window through Sony's released pretrained SampleID model;
4. compares the embeddings; and
5. returns the strongest candidate source ↔ derivative timestamp pairs.

This is **candidate localization**, not a final Nature of Use determination or legal-grade boundary detector.

## Simple upload interface

### Windows setup

Sony's inference checkpoint references internal modules under `sampleid/src`. A normal pip install can omit those source folders, so on Windows use the included editable-install helper:

```powershell
cd C:\Users\saral\projects\sony_sampleid
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

Then launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This clones Sony's repository into `vendor/sampleid` and installs it in editable mode so Hydra can resolve the internal network classes used by the published checkpoint.

## Command line option

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

A smaller hop gives denser candidate timestamps but does **not** by itself make the boundaries more accurate.

## Important limitations

- Sony's released inference model compares chunks; it does not directly return exact sample boundaries.
- Similarity scores need to be calibrated on Lastrada-approved song pairs before using any fixed threshold.
- This POC does not calculate clearance conclusions, prominence, instrument identity, or legal/business split recommendations.
- Do not commit client audio to this repository.
