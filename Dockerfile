FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_ACCEPT_LICENSE=true
ENV PYTHONWARNINGS=ignore::DeprecationWarning
ENV CONDA_OVERRIDE_CUDA=12.0

RUN conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /app
COPY . /app
COPY environment.yml /app/environment.yml

RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f environment.yml

COPY start.sh /app/start.sh
RUN chmod 500 /app/start.sh

ENTRYPOINT ["/app/start.sh"]

EXPOSE 8080 3000
