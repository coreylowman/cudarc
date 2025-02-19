# Generating cuda bindings

Easiest way to do this is to use [bindgen](https://rust-lang.github.io/rust-bindgen/), and use docker images.

## Docker images

You can download official nvidia images from here: https://hub.docker.com/r/nvidia/cuda/tags

Look for tags with the filter: `<cuda-version>-cudnn-devel-ubuntu-<ubuntu version>`. This will have cuda & cudnn headers.

Here is the list of images we currently use to generate bindings:
- nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
- nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
- nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
- nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
- nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
- nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
- nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
- nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04
- nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
- nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
- nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
- nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
- nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

## Running bindgen

Launch whatever docker image you want to generate bindings for:

```bash
cd cudarc
docker run -it -v .:/cudarc -w /cudarc <image tag> /bin/bash
```

This command will bind your current working directory to the cudarc directory in the image.

Then once inside you can run:

```bash
bash install-bindgen.sh && . "$HOME/.cargo/env" && bash run-bindgen.sh
```
