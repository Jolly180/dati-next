import pygame
from random import randint

pygame.init()
width =  1200
height = 800

screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

font = pygame.font.SysFont("Lucida", 32)

radius = 20

balls = []
for b in range(2050):
    balls.append(
        {
            "x": randint(12, (width - 12)),
            "y": randint(12, height - 12),
            "vx": randint(-5, 5),
            "vy": randint(-5, 5),
            "color": (randint(0,255), randint(0,255), randint(0,255))
        }
    )

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    # screen.fill((250, 250, 250))

    for b in balls:
        circle = pygame.draw.circle(screen, b["color"], (b["x"], b["y"]), radius)
        if circle.right >= width or circle.left <= 0:
            b["vx"] *= -1
        if circle.top <= 0 or circle.bottom >= height:
            b["vy"] *= -1
        b["x"] += b["vx"]
        b["y"] += b["vy"]
    t = clock.tick(60)
    fps = round(1000 / t, 1)
    message = font.render(f"FPS: {fps}", True, (10, 10, 10))
    message_rect = message.get_rect(topleft=(20,20))
    screen.blit(message, message_rect)
    pygame.display.update()
