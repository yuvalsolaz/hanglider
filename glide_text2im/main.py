import sys
import torch as th
from visualize import image_show_pil, image_show_pygame


from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# Sampling parameters
batch_size = 1
guidance_scale = 3.0

# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

'''
import matplotlib.pyplot as plt
def update_image(batch, caption=''):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.reshape([batch.shape[2], -1, 3])
    # im = Image.fromarray(reshaped.numpy())
    im = reshaped.numpy()
    plt.imshow(im)
    plt.show()
'''

def generate_image(prompt:str):
    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        image = th.cat([eps, rest], dim=1)
        # image_show_pygame(eps, caption=f'frame #{N}')
        image_show_pygame(rest, caption=f'shape {rest.shape}')
        return image

    # Load / Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
        model.to(device)
        model.load_state_dict(load_checkpoint('base', device))

    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Load / Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27'  # use 27 diffusion steps for very fast sampling

    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
       model_up.convert_to_fp16()
       model_up.to(device)
       model_up.load_state_dict(load_checkpoint('upsample', device))

    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    ##############################
    # Sample from the base model #
    ##############################

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options['text_ctx'])

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()
    return samples

if __name__ == '__main__':
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')
    print(f'using: {device} device\n')

    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <prompt>')
        exit(1)

    prompt = ' '.join(sys.argv[1:])
    print (f'generating image for: {prompt}')
    samples = generate_image(prompt=prompt)

    # Show the output
    image_show_pil(samples, caption=prompt)
    pass
