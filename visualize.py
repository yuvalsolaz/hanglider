
from PIL import Image
import torch as th
import pygame
display_size = 512
h, w = display_size,2.0 * display_size # 64,64
border = 10

pygame.init()
screen = pygame.display.set_mode((w + (2 * border), h + (2 * border)))

def create_pil_image(batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    npimage = reshaped.numpy()
    im = Image.fromarray(npimage)
    return im.resize(size=(512,512))



def image_show_pil(batch: th.Tensor, caption=''):
    create_pil_image(batch).show(title=caption)

def image_show_pygame(batch, caption=''):
    pygame.display.set_caption(caption)
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    # reshaped = scaled.permute(0, 2, 3, 1).reshape([batch.shape[2], -1, 3])
    reshaped0 = scaled.permute(0, 3, 2, 1)[0].reshape([batch.shape[2], -1, 3])
    reshaped1 = scaled.permute(0, 3, 2, 1)[1].reshape([batch.shape[2], -1, 3])
    npimage0 = reshaped0.numpy()
    npimage1 = reshaped1.numpy()
    # Clear screen to white before drawing
    screen.fill((0, 0, 0))

    # Convert to a surface and splat onto screen offset by border width and height
    surface0 = pygame.surfarray.make_surface(npimage0)
    surface1 = pygame.surfarray.make_surface(npimage1)
    scaled_surface0 = pygame.transform.smoothscale(surface=surface0, size=(display_size,display_size),)
    scaled_surface1 = pygame.transform.smoothscale(surface=surface1, size=(display_size,display_size))
    screen.blit(scaled_surface0, (border, border))
    screen.blit(scaled_surface1, (display_size+2*border, border))

    pygame.display.flip()
