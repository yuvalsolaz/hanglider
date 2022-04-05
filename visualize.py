from PIL import Image
import torch as th
import pygame

display_size = 512
h, w = display_size, display_size
border = 10

pygame.init()
screen = pygame.display.set_mode((w + (2 * border), h + (2 * border)))


def create_pil_image(batch: th.Tensor):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    npimage = reshaped.numpy()
    im = Image.fromarray(npimage)
    return im.resize(size=(512, 512))


def image_show_pil(batch: th.Tensor, caption=''):
    create_pil_image(batch).show(title=caption)


def image_show_pygame(batch_tuple: th.Tensor, caption=''):
    pygame.display.set_caption(caption)
    batch = batch_tuple[0]
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(1 ,2 ,0).reshape([batch.shape[2], -1, 3])
    npimage = reshaped.numpy()

    # Convert to a surface and splat onto screen offset by border width and height
    surface = pygame.surfarray.make_surface(npimage)
    scaled_surface = pygame.transform.smoothscale(surface=surface, size=(display_size, display_size), )
    rotate_surface = pygame.transform.rotate(surface=scaled_surface, angle=-90.0)

    screen.blit(rotate_surface, (border, border))
    pygame.display.flip()
