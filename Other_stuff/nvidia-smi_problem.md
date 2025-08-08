# How to solve the problem "... couldn't communicate with the NVIDIA driver..." when running nvidia-smi?

when running `nvidia-smi` in the terminal, you may encounter the following error:
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running. 
```
It may be caused by the following reasons:
1. The BIOS setting is not correct. You need to disable the Secure Boot in the BIOS setting.
REF https://gist.github.com/espoirMur/65cec3d67e0a96e270860c9c276ab9fa
2. The ubunut kernal is not compatible with the NVIDIA driver. You need to update the kernal to the specific version as the NVIDIA driver installed. [CLICK HERE](ubuntu-kernel-switching.md)
