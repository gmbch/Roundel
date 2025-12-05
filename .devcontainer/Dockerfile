FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir \
    matplotlib==3.5.3 \
    nibabel==5.2.1 \
    numpy==1.22.4 \
    pandas==2.0.3 \
    Pillow==9.2.0 \
    "protobuf>=3.20,<6" \
    pydicom==2.2.2 \
    pylibjpeg==1.4.0 \
    pylibjpeg-libjpeg==1.3.4 \
    pylibjpeg-openjpeg==1.3.2 \
    scikit-image==0.20.0 \
    scikit-learn==1.2.2 \
    scipy==1.9.0 \
    tqdm==4.64.0 \
    opencv-python-headless==4.7.0.68 \
    streamlit==1.40.1 \
    streamlit_drawable_canvas \
    imageio

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
