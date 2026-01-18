import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem
import os
import requests
import zipfile
import io
import shutil
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz as _scipy_load_npz


# Helper: try to download and extract a zip URL to the repository root
def _fetch_and_extract(url: str, dest: str = ".") -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        # If zip, extract in-memory
        if url.lower().endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            z.extractall(dest)
            return True
        # Otherwise save to file
        filename = os.path.join(dest, os.path.basename(url))
        with open(filename, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        return True
    except Exception:
        return False


# Paths
cleaned_data_path = "data/cleaned_data.csv"
transformed_data_path = "data/transformed_data.npz"
track_ids_path = "data/track_ids.npy"
filtered_data_path = "data/collab_filtered_data.csv"
interaction_matrix_path = "data/interaction_matrix.npz"
transformed_hybrid_data_path = "data/transformed_hybrid_data.npz"


# Ensure required files exist or attempt to fetch them if a DATA_URL is provided
missing_files = []
for p in [cleaned_data_path, transformed_data_path, track_ids_path, filtered_data_path, interaction_matrix_path, transformed_hybrid_data_path]:
    if not os.path.exists(p):
        missing_files.append(p)

if missing_files:
    data_url = os.environ.get("DATA_URL") or (st.secrets.get("DATA_URL") if hasattr(st, "secrets") else None)
    if data_url:
        st.info("Required data files are missing. Attempting to download data from DATA_URL...")
        ok = _fetch_and_extract(data_url, dest=".")
        if ok:
            st.success("Download complete — continuing startup.")
        else:
            st.error("Failed to download data from DATA_URL. Please provide the `data/` folder via Git LFS or a valid ZIP URL in the `DATA_URL` env var.")
            st.stop()
    else:
        st.error("Required data files are missing: {}.\nProvide the `data/` folder (tracked with Git LFS) or set the `DATA_URL` environment variable to a ZIP containing the `data/` directory.".format(", ".join(missing_files)))
        st.stop()

# load the data
songs_data = pd.read_csv(cleaned_data_path)

# load the transformed data
def _load_sparse_matrix(path: str):
    """Load a sparse matrix from a .npz file.
    Tries scipy.sparse.load_npz first, then falls back to loading with numpy
    and reconstructing a CSR matrix from common save formats.
    """
    try:
        return _scipy_load_npz(path)
    except Exception:
        pass

    try:
        with np.load(path, allow_pickle=True) as npz:
            files = list(npz.files)
            # common single-array save (arr_0)
            if "arr_0" in npz:
                return csr_matrix(npz["arr_0"][()]) if npz["arr_0"].dtype == object else csr_matrix(npz["arr_0"])

            # scipy.save_npz stores 'data','indices','indptr','shape'
            if set(["data", "indices", "indptr", "shape"]).issubset(files):
                data = npz["data"]
                indices = npz["indices"]
                indptr = npz["indptr"]
                shape = tuple(npz["shape"])
                return csr_matrix((data, indices, indptr), shape)

            # single unnamed array
            if len(files) == 1:
                return csr_matrix(npz[files[0]])

    except Exception:
        pass

    raise ValueError(f"Could not load sparse matrix from {path}")

transformed_data = _load_sparse_matrix(transformed_data_path)

# load the track ids
track_ids = load(track_ids_path, allow_pickle=True)

# load the filtered songs data
filtered_data = pd.read_csv(filtered_data_path)

# load the interaction matrix
interaction_matrix = _load_sparse_matrix(interaction_matrix_path)

# load the transformed hybrid data
transformed_hybrid_data = _load_sparse_matrix(transformed_hybrid_data_path)


# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)
# artist name
artist_name = st.text_input('Enter the artist name:')
st.write('You entered:', artist_name)
# lowercase the input
song_name = song_name.lower()
artist_name = artist_name.lower()

# k recommndations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)

if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():   
    # type of filtering
    filtering_type = "Hybrid Recommender System"

    # diversity slider
    diversity = st.slider(label="Diversity in Recommendations",
                        min_value=1,
                        max_value=9,
                        value=5,
                        step=1)

    content_based_weight = 1 - (diversity / 10)
    
    # plot a bar graph
    chart_data = pd.DataFrame({
        "type" : ["Personalized", "Diverse"],
        "ratio": [10 - diversity, diversity]
    })
    
    st.bar_chart(chart_data,x="type",y="ratio")
    
else:
    # type of filtering
    filtering_type = 'Content-Based Filtering'

# Button
if filtering_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):
        if ((songs_data["name"] == song_name) & (songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = content_recommendation(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=songs_data,
                                                     transformed_data=transformed_data,
                                                     k=k)
            
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
            
elif filtering_type == "Hybrid Recommender System":
    if st.button('Get Recommendations'):
        st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
        recommender = HybridRecommenderSystem(
                                                number_of_recommendations= k,
                                                weight_content_based= content_based_weight
                                                )
                                
        # get the recommendations
        recommendations = recommender.give_recommendations(song_name= song_name,
                                                        artist_name= artist_name,
                                                        songs_data= filtered_data,
                                                        transformed_matrix= transformed_hybrid_data,
                                                        track_ids= track_ids,
                                                        interaction_matrix= interaction_matrix)
        # Display Recommendations
        for ind , recommendation in recommendations.iterrows():
            song_name = recommendation['name'].title()
            artist_name = recommendation['artist'].title()
            
            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"#### **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            elif ind == 1:   
                st.markdown("### Next Up 🎵")
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')