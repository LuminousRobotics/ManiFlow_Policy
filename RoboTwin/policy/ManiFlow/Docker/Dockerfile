FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/lib/nvidia"
ENV PYOPENGL_PLATFORM egl
ENV MUJOCO_GL=egl

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    zip \
    unzip \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    build-essential \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Vulkan related packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends rsync tree \
    swig unzip htop tmux bash-completion \
    libssl-dev libcurl4-openssl-dev zlib1g-dev qtbase5-dev qtdeclarative5-dev \
    libvulkan1 libosmesa6 \
    mesa-vulkan-drivers vulkan-tools \
    libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# Install system packages
RUN apt-get update && apt-get install -y \
    nano \
    vim \
    pkg-config \
    cmake \
    xvfb \
    xorg-dev \
    ninja-build \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

SHELL ["/bin/bash", "-c"]
RUN echo "cd /root" >> /root/.bashrc

RUN pip install --upgrade pip

WORKDIR /workspace

# install python packages from requirements.txt
COPY requirements.txt /root/
RUN pip install -r /root/requirements.txt

# Install Pytorch3D
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install flash-attn
RUN MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation

CMD ["bash"]