## DDPM Example
Example for DDPM

## Train model
```python
python train.py
```

## Inference
```python
ddpm = DDPM(...)
out = ddpm.sample(image_size=32, batch_size=16)
```

## References
1. [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)