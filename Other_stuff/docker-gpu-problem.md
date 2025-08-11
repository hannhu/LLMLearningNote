# Docker GPU Access Error Fix

**Error**: `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Updated**: March 26, 2025

## Problem Description

When starting a Docker container on a GPU cloud server with the command `docker run --gpus all [image_name]`, you may encounter the following error if NVIDIA Container Toolkit is not installed:

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

## Root Cause

NVIDIA Container Toolkit is the tool that enables Docker to access GPU resources. After installing Docker on a GPU cloud server, if NVIDIA Container Toolkit is not installed, Docker cannot select GPU devices, resulting in the above error.

## Solution

### 1. Verify NVIDIA GPU Driver Installation

First, confirm that the NVIDIA GPU driver is installed on your GPU instance:

```bash
nvidia-smi
```

> **Note**: GPU instances do not come with drivers pre-installed. You need to install the appropriate driver separately. If NVIDIA GPU driver is not installed, Docker cannot access GPU devices.

If the command displays the driver version, it means the NVIDIA GPU driver is successfully installed. Otherwise, continue to install Tesla driver or GRID driver.

### 2. Verify Docker Installation

Confirm that Docker is installed on your GPU instance:

```bash
sudo docker -v
```

If the command displays the Docker version, it means Docker is installed. Otherwise, please install Docker first.

### 3. Install NVIDIA Container Toolkit

The following steps cover CentOS, Alibaba Cloud Linux, and Ubuntu. For other operating systems, please refer to [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### For CentOS/Alibaba Cloud Linux:

```bash
# Configure repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install
sudo yum install -y nvidia-container-toolkit

# Restart Docker service
sudo systemctl restart docker
```

#### For Ubuntu:

```bash
# Configure repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

# Install
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker service
sudo systemctl restart docker
```

### 4. Verify Installation

Check that NVIDIA Container Toolkit has been successfully installed:

#### For CentOS/Alibaba Cloud Linux:
```bash
sudo rpm -qa | grep nvidia-container-toolkit
```

#### For Ubuntu:
```bash
sudo dpkg -l | grep nvidia-container-toolkit
```

If the command displays the NVIDIA Container Toolkit version, it means NVIDIA Container Toolkit has been correctly installed.

## Testing

After installation, test GPU access with Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

This should successfully display GPU information within the container.