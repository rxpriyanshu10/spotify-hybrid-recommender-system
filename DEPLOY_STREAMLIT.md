# Deploying this repo to Streamlit Cloud (using Git LFS)

This project already contains a Streamlit app in `app.py`. These instructions show how to use Git LFS for large model/data files and deploy on Streamlit Cloud.

1) Install Git LFS locally

```bash
# macOS / Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
# Windows: install via Git for Windows installer or https://git-lfs.github.com/

# initialize
git lfs install
```

2) Ensure `.gitattributes` exists (it was added) and contains tracked patterns (models, data, joblib, npz, npy files). If you need to track additional patterns run:

```bash
# example tracking commands (optional)
git lfs track "transformer.joblib"
git lfs track "data/**"
git add .gitattributes
```

3) Stage, commit, and push LFS-tracked files

```bash
git add .
git commit -m "Add Git LFS tracking and Streamlit wrapper"
git push origin main
```

Note: Git LFS stores pointers in Git and uploads the real blobs to LFS storage. GitHub provides limited free LFS storage; consider enabling billing if your dataset/model is large.

4) Deploy on Streamlit Cloud

- Go to https://streamlit.io/cloud
- Connect your GitHub account and select this repository
- Choose the branch (e.g., `main`) and set the main file to `streamlit_app.py` (or `app.py`)
- Deploy. Streamlit Cloud will run `pip install -r requirements.txt` automatically.

5) Troubleshooting & tips

- If builds fail due to memory/time, reduce model/data size or load artifacts from an external storage (S3, GitHub Releases).
- If GitHub LFS bandwidth/storage is exceeded, consider hosting large artifacts externally and loading them at runtime.

If you want, I can run the exact Git LFS commands locally and prepare the commit for you. Tell me whether you want me to initialize LFS and commit now.