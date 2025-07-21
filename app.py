import streamlit as st
import nbformat
from nbconvert import PythonExporter
import requests

# Konfigurasi halaman
st.set_page_config(page_title="Notebook Viewer", layout="wide")

st.title("📘 Streamlit Viewer untuk Notebook Jupyter dari GitHub")

# Input URL notebook dari GitHub
notebook_url = st.text_input("Masukkan URL file notebook dari GitHub (.ipynb):", 
                             "https://raw.githubusercontent.com/username/repo-name/main/Copy_of_FIX_SCRIPT_2.ipynb")

if notebook_url:
    try:
        # Ambil konten dari GitHub (pastikan URL mentah/raw)
        response = requests.get(notebook_url)
        response.raise_for_status()

        # Parse notebook
        notebook = nbformat.reads(response.text, as_version=4)

        # Konversi ke kode Python
        exporter = PythonExporter()
        source_code, _ = exporter.from_notebook_node(notebook)

        # Tampilkan isi notebook sebagai kode
        st.subheader("📜 Source Code dari Notebook:")
        st.code(source_code, language="python")

        # Tampilkan seluruh sel markdown (jika ingin)
        st.subheader("📝 Markdown (Komentar dan Penjelasan):")
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                st.markdown(cell['source'])

    except Exception as e:
        st.error(f"Gagal memuat notebook: {e}")
