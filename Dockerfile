# Use an official Python runtime as the base image
# FROM python:3.12-slim
# Base image with GPU support
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Set CUDA and NVIDIA library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib:${LD_LIBRARY_PATH}

RUN ldconfig

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Set environment variables for headless rendering with EGL
ENV PYOPENGL_PLATFORM=egl
ENV DISPLAY=:99

RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox

# Install software-properties-common first to enable adding PPAs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install system-level dependencies, including Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    bison \
    flex \
    libncurses5-dev \
    libncursesw5-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    git \
    wget \
    unzip \
    ca-certificates \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1 \
    libglu1-mesa \
    mesa-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py



# Set the working directory inside the container
WORKDIR /hyperagents

# Copy the entire repository into the container
COPY . .

# Robust runtime fix for libcuda.so (covers /usr/lib64 and RO mounts).
COPY docker/fix-cuda.sh /usr/local/bin/fix-cuda.sh
RUN chmod +x /usr/local/bin/fix-cuda.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install proofgrader for imo_proof domain when that optional checkout is present.
RUN if [ -d proofgrader_repo ]; then pip install -e proofgrader_repo; fi

# Download things for balrog domains
RUN python -m domains.balrog.scripts.post_install

# For Genesis: install PyTorch with CUDA support
# First check the Cuda version: nvidia-smi
# If Cuda version is 11.8:
# RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# If Cuda version is 12.1:
# RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# If Cuda version is 12.4:
# RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url  https://download.pytorch.org/whl/cu124
# If Cuda version is 13.0:
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Ensure the libcuda symlink fix runs before your command
ENTRYPOINT ["/usr/local/bin/fix-cuda.sh"]

# Keep the container running by default
CMD ["tail", "-f", "/dev/null"]
