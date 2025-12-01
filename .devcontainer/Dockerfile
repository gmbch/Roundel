# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.0/containers/python-3/.devcontainer/base.Dockerfile
 
# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
FROM tensorflow/tensorflow:2.9.1-gpu
RUN apt-get install libopenexr-dev -y
RUN pip install importlib-metadata==3.10.1
RUN pip install importlib-resources==5.9.0
RUN pip install json5==0.9.28
RUN pip install jsonref==1.1.0
RUN pip install jsonschema==4.20.0
RUN pip install jsonschema-specifications==2023.11.2
RUN pip install keras==2.9.0
RUN pip install Keras-Preprocessing==1.1.2
RUN pip install matplotlib==3.5.3
RUN pip install matplotlib-inline==0.1.7
RUN pip install neptune==1.8.6
RUN pip install neptune-tensorflow-keras==2.2.2
RUN pip install nibabel==5.2.1
RUN pip install numpy==1.22.4
RUN pip install oauthlib==3.2.0
RUN pip install OpenEXR==1.3.8
RUN pip install openpyxl==3.1.5
RUN pip install pandas==2.0.3
RUN pip install Pillow==9.2.0
RUN pip install plotly==5.10.0
RUN pip install protobuf==3.19.4
RUN pip install pydicom==2.2.2
RUN pip install pylibjpeg==1.4.0
RUN pip install pylibjpeg-libjpeg==1.3.4
RUN pip install pylibjpeg-openjpeg==1.3.2
RUN pip install rasterio==1.3.11
RUN pip install scikit-image==0.20.0
RUN pip install scikit-learn==1.2.2
RUN pip install scipy==1.9.0
RUN pip install seaborn==0.13.0
RUN pip install statsmodels==0.14.1
RUN pip install tensorflow==2.9.1
RUN pip install tensorflow-datasets==4.6.0
RUN pip install tensorflow-estimator==2.9.0
RUN pip install tensorflow-graphics==2021.12.3
RUN pip install tensorflow-io==0.26.0
RUN pip install tensorflow-io-gcs-filesystem==0.26.0
RUN pip install tensorflow-metadata==1.9.0
RUN pip install tqdm==4.64.0
RUN pip install volumentations==0.1.8
RUN pip install Shapely==1.8.1.post1
RUN pip install opencv-python-headless==4.7.0.68
RUN pip install ipykernel
RUN pip install albumentations
RUN pip install streamlit
RUN pip install streamlit_drawable_canvas
RUN pip install lifelines
RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

# # Create non-root user.
# ARG USERNAME=tina
# ARG USER_UID=1001
# ARG USER_GID=$USER_UID
 
# RUN groupadd --gid $USER_GID $USERNAME && \
#     useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
#     # Add user to sudoers.
#     apt-get update && \
#     apt-get install -y sudo && \
#     echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
#     chmod 0440 /etc/sudoers.d/$USERNAME && \
#     # Change default shell to bash.
#     usermod --shell /bin/bash $USERNAME