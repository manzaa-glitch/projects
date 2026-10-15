# Applied AI — DTW / librosa refinement POC

This is **Step 2** after Sony SampleID.

Sony gives us a rough candidate such as:

- source: **0:42.5–0:47.5**
- derivative: **1:10.0–1:15.0**

This POC uses **librosa subsequence Dynamic Time Warping (DTW)** to search around
the derivative candidate and tighten where the supplied source excerpt best aligns.

## Why this exists

Sony embeddings are useful for candidate localization, but 5-second windows are not exact
clearance timestamps. DTW gives us a more fine-grained time alignment once we already
have a plausible region.

## Important limitation

This version treats the Sony **source start/end as the complete query excerpt**. It can
refine the derivative placement, but it does **not yet** infer that only 2 seconds inside
a rough 5-second source window were actually sampled.

Also, chroma intentionally throws away much timbre detail. That helps it survive pitch,
instrumentation and mix changes, but means it is not safe as a standalone sample detector.

## Run locally

```bash
cd dtw_refinement
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Mac/Linux activation:

```bash
source .venv/bin/activate
```

## Run in Google Colab

Open:

https://colab.research.google.com/github/manzaa-glitch/projects/blob/main/dtw_refinement/Applied_AI_DTW_Refinement_Colab.ipynb

The Colab notebook requires no Sony checkpoint. Upload both songs, paste one Sony candidate,
and run the refinement.

## Output

The refiner returns:

- source excerpt supplied
- Sony derivative hint
- refined derivative start/end
- matched duration
- estimated pitch shift in semitones
- derivative/source duration ratio
- mean DTW alignment cost

**Mean DTW cost is not a probability. Lower means the two sequences aligned more closely
under the selected features.**
