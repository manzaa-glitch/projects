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


## Run in Google Colab (recommended for iPad)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manzaa-glitch/projects/blob/main/sony_sampleid/Applied_AI_SampleID_Colab.ipynb)

On an iPad, open the Colab link above and run each cell from top to bottom. The notebook will let you upload the source song and derivative song, run Sony SampleID in the browser, display matching timestamp windows, and download a CSV.

## Simple upload interface

Sony currently declares Python **3.12+**.

```bash
cd sony_sampleid
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app.py
```

Then:
1. upload the original/source song;
2. upload the sampled/derivative song;
3. click **Analyze**;
4. review the strongest candidate timestamp matches; and
5. download the results as CSV if needed.

The first run may take longer because Sony's published pretrained checkpoint is downloaded automatically.

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
