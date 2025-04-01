# Use the PyTorch base image with CUDA 12.4 and cuDNN 9
# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y     wget     git     && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Ensure Conda is available (it is already installed in the PyTorch image)
# ENV PATH=/opt/conda/bin:/home/prl/.vscode-server/cli/servers/Stable-e54c774e0add60467559eb0d1e229c6452cf8447/server/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

# # Copy the Conda environment file
# COPY src/environment_linux.yml /workspace/environment.yml

# # Update the existing Conda environment instead of creating a new one
# RUN conda env update --name base --file /workspace/environment.yml
