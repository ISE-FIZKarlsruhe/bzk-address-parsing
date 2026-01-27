# Use ubuntu for main image, installing C libraries on top of other images is often more complicated
FROM ubuntu:22.04 


# Install build dependencies
RUN apt update && apt install -y curl build-essential autoconf automake libtool pkg-config git wget

# Install and configure Miniconda
RUN mkdir -p /miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    rm /miniconda3/miniconda.sh
RUN /miniconda3/bin/conda init bash
ENV PATH="/miniconda3/bin:${PATH}"
RUN conda config --set always_yes yes
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda update -q conda

# Install JupyterLab
RUN conda install conda-forge::jupyterlab

# Install libpostal and python bindings
RUN cd / && git clone https://github.com/openvenues/libpostal && \
    cd libpostal && \
    ./bootstrap.sh && \
    ./configure --datadir=/data/libpostal && \
    make && \
    make install && \
    ldconfig
RUN pip install postal
WORKDIR /workspace
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser"]