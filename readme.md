# Pseudo Numerical Methods for Diffusion Models on Manifolds (PNDM, PLMS | ICLR2022)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pseudo-numerical-methods-for-diffusion-models-1/image-generation-on-celeba-64x64)](https://paperswithcode.com/sota/image-generation-on-celeba-64x64?p=pseudo-numerical-methods-for-diffusion-models-1)

This repo is the official PyTorch implementation for the paper [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://openreview.net/forum?id=PlKWVd2yBkY) (PNDM, PLMS | ICLR2022)

by [Luping Liu](https://luping-liu.github.io/), [Yi Ren](https://rayeren.github.io/), Zhijie Lin, Zhou Zhao (Zhejiang University).


## What does this code do?
This code is not only the official implementation for PNDM, but also a generic framework for DDIM-like models including:
- [x] [Pseudo Numerical Methods for Diffusion Models on Manifolds (PNDM)](https://openreview.net/forum?id=PlKWVd2yBkY)
- [x] [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)
- [x] [Score-Based Generative Modeling through Stochastic Differential Equations (PF)](https://arxiv.org/abs/2011.13456)
- [x] [Improved Denoising Diffusion Probabilistic Models (iDDPM)](https://arxiv.org/abs/2102.09672)

### Structure
This code contains three main objects including method, schedule and model. The following table shows the options 
supported by this code and the role of each object.

| Object   | Option                        | Role                                          |
|----------|-------------------------------|-----------------------------------------------|
| method   | DDIM, S-PNDM, F-PNDM, FON, PF | the numerical method used to generate samples |
| schedule | linear, quad, cosine          | the schedule of adding noise to images        |
| model    | DDIM, iDDPM, PF, PF_deep      | the neural network used to fit noise          |

All of them can be combined at will, so this code provide at least 5x3x4=60 choices to generate samples.


## Integration with 🤗 Diffusers library

PNDM is now also available in 🧨 Diffusers and accesible via the [PNDMPipeline](https://huggingface.co/docs/diffusers/api/pipelines/pndm).
Diffusers allows you to test PNDM in PyTorch in just a couple lines of code.

You can install diffusers as follows:

```
pip install diffusers torch accelerate
```

And then try out the sampler/scheduler with just a couple lines of code:

```python
from diffusers import PNDMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
pndm = PNDMPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = pndm(num_inference_steps=50).images[0]

# save image
image.save("pndm_generated_image.png")
```

The PNDM scheduler can also be used with more powerful diffusion models such as [Stable Diffusion](https://huggingface.co/docs/diffusers/v0.7.0/en/api/pipelines/stable_diffusion#stable-diffusion-pipelines)

You simply need to [accept the license on the Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5), login with `huggingface-cli login` and install transformers:

```
pip install transformers
```

Then you can run:

```python
from diffusers import StableDiffusionPipeline, PNDMScheduler

pndm = PNDMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=pndm)

image = pipeline("An astronaut riding a horse.").images[0]
image.save("astronaut_riding_a_horse.png")
```


## How to run the code

### Dependencies
Run the following to install a subset of necessary python packages for our code.
```bash
pip install -r requirements.txt
```
Tip: mpi4py can make the generation process faster using multi-gpus. It is not necessary and can be removed freely.

### Usage
Evaluate our models through main.py.
```bash
python main.py --runner sample --method F-PNDM --sample_speed 50 --device cuda --config ddim_cifar10.yml --image_path temp/results --model_path temp/models/ddim/ema_cifar10.ckpt
```
- runner (train|sample): choose the mode of runner 
- method (DDIM|FON|S-PNDM|F-PNDM|PF): choose the numerical methods
- sample_speed: control the total generation step
- device (cpu|cuda:0): choose the device to use
- config: choose the config file
- image_path: choose the path to save images
- model_path: choose the path of model

Train our models through main.py.
```bash
python main.py --runner train --device cuda --config ddim_cifar10.yml --train_path temp/train
```
- train_path: choose the path to save training status

### Checkpoints & statistics
All checkpoints of models and precalculated statistics for FID are provided in this [Onedrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/3170105432_zju_edu_cn/EhjaZe0ZhnxOrPvejWp0f-cBv8F0xOL9J8xaVyor0fLZEA).


## References
If you find the code useful for your research, please consider citing:
```bib
@inproceedings{liu2022pseudo,
    title={Pseudo Numerical Methods for Diffusion Models on Manifolds},
    author={Luping Liu and Yi Ren and Zhijie Lin and Zhou Zhao},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=PlKWVd2yBkY}
}
```
This work is built upon some previous papers which might also interest you:
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
- Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. International Conference on Learning Representations. 2021.
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. International Conference on Learning Representations. 2021.

