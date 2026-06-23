# Deploying & submitting

This covers three things: refreshing an existing local copy, pushing the code to
GitHub, and putting the sandbox demo online for the public link the challenge
requires.

---

## 0. Already set this up earlier? You do NOT need to start over.

If you previously extracted an older copy and ran it, the only things that
changed in the latest version are the **optional** LLM reasoning path (the
`.env` loader and the Gemini default model). **The core ranking is unchanged**,
so a `submission.csv` you produced with the plain command is still valid and
identical to what the new code produces.

You only need the newer code if you want the Gemini-polished reasoning. To
refresh without redoing everything:

1. **Keep your big data file.** Move `candidates.jsonl` somewhere safe — you do
   not need to re-download or re-extract it.
2. **Swap in the new code.** Delete (or rename) your old `redrob-ranker` folder
   and extract the new `redrob-ranker.zip` in its place.
3. **Put the data back.** Copy `candidates.jsonl` into the new folder.
4. **Reuse your environment.** Activate the same virtual environment you already
   made (`.venv\Scripts\activate` on Windows). The dependencies are unchanged,
   so there's nothing new to install. (If you'd rather start clean, recreate it —
   see step 1 of the README quickstart; it only takes a minute.)
5. Add your key once (see README → "Configuring a free LLM") and run.

Nothing about the local run needs to be done "from the beginning."

---

## 1. Push the code to GitHub

Prerequisites: Git installed, a GitHub account.

1. On github.com, create a new **empty** repository named `redrob-ranker`
   (don't add a README/license — the repo already has them).
2. In a terminal, from inside the `redrob-ranker` folder:

```bash
git init
git add .
git commit -m "Redrob intelligent candidate ranker"
git branch -M main
git remote add origin https://github.com/<your-username>/redrob-ranker.git
git push -u origin main
```

The `.gitignore` already keeps the large `candidates.jsonl` and your private
`.env` out of the commit. After pushing, open the repo on GitHub and confirm
neither of those got uploaded. Put the repo URL into `submission_metadata.yaml`
under `github_repo`.

---

## 2. Put the sandbox online (the public demo link)

Pick ONE. Option A is the easiest and free.

### Option A — Streamlit Community Cloud (recommended)

1. Go to share.streamlit.io and sign in with your GitHub account.
2. Click **Create app** → **Use existing repo**.
3. Set: **Repository** = your `redrob-ranker` repo, **Branch** = `main`,
   **Main file path** = `app/streamlit_app.py`.
4. (Optional) choose a custom app URL.
5. Click **Deploy**. In a couple of minutes you'll get a public
   `https://<name>.streamlit.app` link that anyone can open.

Notes:
- No dependency changes needed. Streamlit Cloud has Streamlit pre-installed and
  installs the rest from the repo's `requirements.txt` (numpy / scipy /
  scikit-learn / pyyaml).
- The app ships with `sample_data/demo_candidates.jsonl`, so it works out of the
  box without uploading the big file. Don't feed the full 100K pool to the hosted
  demo — free instances are memory-limited and the sandbox only needs ≤100.
- To enable Gemini reasoning in the hosted demo (optional), add `LLM_PROVIDER`,
  `GEMINI_API_KEY`, and `LLM_MODEL` under the app's **Settings → Secrets** (these
  stay private, never in the repo).

### Option B — Hugging Face Spaces (Docker)

1. On huggingface.co, **New Space** → choose the **Docker** SDK.
2. Push this repo to the Space (it includes a `Dockerfile`). HF builds the image
   and serves the Streamlit app. Free CPU tier; the Space sleeps after a period
   of inactivity and wakes on the next visit.

### Option C — Local Docker (the "docker run recipe" the spec also accepts)

```bash
docker build -t redrob-ranker .
docker run -p 8501:8501 redrob-ranker
# open http://localhost:8501
```

This is local only. For a link others can open, use Option A or B.

Put the chosen public URL into `submission_metadata.yaml` under `sandbox_link`.

---

## 3. Final submission checklist

- [ ] Rename the output CSV to your participant ID: `<participant_id>.csv`
      (the portal keys on the filename). Re-run `validate_submission.py` on it.
- [ ] Fill every `TODO` in `submission_metadata.yaml` (team, contact,
      `github_repo`, `sandbox_link`, AI-tools declaration).
- [ ] Code pushed to GitHub (public), with no `candidates.jsonl` and no `.env`.
- [ ] Sandbox deployed and the link opens for a logged-out visitor.
- [ ] Deck PDF ready (`redrob_approach_deck.pdf`).
- [ ] Submit on the portal: CSV + metadata + GitHub URL + sandbox link + deck.
