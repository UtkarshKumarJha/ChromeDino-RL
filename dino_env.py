import os
import pygame
import numpy as np
import gymnasium as gym
import random


pygame.init()

SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [
    pygame.image.load(os.path.join("assets/Dino", "DinoRun1.png")),
    pygame.image.load(os.path.join("assets/Dino", "DinoRun2.png")),
]
JUMPING = pygame.image.load(os.path.join("assets/Dino", "DinoJump.png"))
DUCKING = [
    pygame.image.load(os.path.join("assets/Dino", "DinoDuck1.png")),
    pygame.image.load(os.path.join("assets/Dino", "DinoDuck2.png")),
]

SMALL_CACTUS = [
    pygame.image.load(os.path.join("assets/Cactus", "SmallCactus1.png")),
    pygame.image.load(os.path.join("assets/Cactus", "SmallCactus2.png")),
    pygame.image.load(os.path.join("assets/Cactus", "SmallCactus3.png")),
]
LARGE_CACTUS = [
    pygame.image.load(os.path.join("assets/Cactus", "LargeCactus1.png")),
    pygame.image.load(os.path.join("assets/Cactus", "LargeCactus2.png")),
    pygame.image.load(os.path.join("assets/Cactus", "LargeCactus3.png")),
]

BIRD = [
    pygame.image.load(os.path.join("assets/Bird", "Bird1.png")),
    pygame.image.load(os.path.join("assets/Bird", "Bird2.png")),
]

CLOUD = pygame.image.load(os.path.join("assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("assets/Other", "Track.png"))

FONT_COLOR=(0,0,0)

class Dinosaur:
    JUMP_VEL = 8.5
    X_POS = 80
    Y_POS = 300
    Y_POS_DUCK = 340

    def __init__(self):
        self.run_img = RUNNING     # list of 2 frames
        self.duck_img = DUCKING    # list of 2 frames
        self.jump_img = JUMPING    # single image

        self.image = self.run_img[0]
        self.rect = self.image.get_rect()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS

        self.state = "run"          # could be "run", "duck", or "jump"
        self.jump_vel = self.JUMP_VEL
        self.step_index = 0

    def update(self, userInput):
        # State transitions
        if (userInput[pygame.K_UP] or userInput[pygame.K_SPACE]) and self.state != "jump":
            self.state = "jump"
        elif userInput[pygame.K_DOWN] and self.state != "jump":
            self.state = "duck"
        elif self.state != "jump":
            self.state = "run"

        # Perform state behavior
        if self.state == "jump":
            self.jump()
        elif self.state == "duck":
            self.duck()
        else:
            self.run()

        # Reset step index for animation
        if self.step_index >= 10:
            self.step_index = 0

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.rect.y = self.Y_POS
        self.step_index += 1

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        self.rect.y -= self.jump_vel * 4
        self.jump_vel -= 0.8

        if self.jump_vel < -self.JUMP_VEL:
            self.jump_vel = self.JUMP_VEL
            self.state = "run"  # Go back to running after jump

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))




class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self,game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self,game_speed):
        self.rect.x -= game_speed

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    BIRD_HEIGHTS = [180]

    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = random.choice(self.BIRD_HEIGHTS)
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1


class DinoEnv(gym.Env):
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = SCREEN

        # Game objects
        self.player = Dinosaur()
        self.obstacles = []
        self.clouds = [Cloud() for _ in range(3)]
        self.bg_x_pos = 0


        # Observation: dino_y, rel_x, obs_y
        self.observation_space = gym.spaces.Box(
        low=np.array([0, -20, 0, 0, 0, -1]),
        high=np.array([SCREEN_HEIGHT, 20, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH, 2]),
        dtype=np.float32
    )


        # Actions: 0 - nothing, 1 - jump, 2 - duck
        self.action_space = gym.spaces.Discrete(3)

        # Constants
        self.game_speed = 20
        self.spawn_timer = 0
        self.points = 0
        self.clock = pygame.time.Clock()

    def _get_obs(self):
        
        self.obstacles = [obs for obs in self.obstacles if obs.rect.right > 0]
        future_obs = [obs for obs in self.obstacles if obs.rect.x >= self.player.rect.right]

        if future_obs:
            next_obs = future_obs[0]
            rel_x = next_obs.rect.x - self.player.rect.x
            obs_y = next_obs.rect.y
            obs_width = next_obs.rect.width
            obs_type = (
                1 if isinstance(next_obs, SmallCactus)
               else 2 if isinstance(next_obs, LargeCactus)
               else 3
            )
        else:
            rel_x = SCREEN_WIDTH
            obs_y = SCREEN_HEIGHT
            obs_width = 0
            obs_type = 0  # no obstacle

        return np.array([
            self.player.rect.y,
            self.player.jump_vel,
            rel_x,
            obs_y,
            obs_width,
            obs_type
        ], dtype=np.float32)


    def spawn_obs(self):
        if not self.obstacles or self.obstacles[-1].rect.x < SCREEN_WIDTH - 300:
            choice = random.randint(0, 2)
            if choice == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif choice == 1:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            else:
                self.obstacles.append(Bird(BIRD))

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.25
        
        user_input = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_SPACE: False}
        if action == 1:
            user_input[pygame.K_UP] = True
        elif action == 2:
            user_input[pygame.K_DOWN] = True
        
        self.player.update(user_input)
        self.spawn_timer += 1
        if self.spawn_timer >= 20:
            self.spawn_obs()
            self.spawn_timer = 0
        for obs in self.obstacles:
            obs.update(self.game_speed)
            if self.player.rect.colliderect(obs.rect):
                reward = -10.0
                terminated = True
        
        self.obstacles = [obs for obs in self.obstacles if obs.rect.x > -obs.rect.width]

        obs = self._get_obs()    

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Dinosaur()
        self.obstacles = []
        self.points = 0
        self.spawn_timer = 0
        self.game_speed = 20
        return self._get_obs(), {}
    def render(self):
        if self.render_mode == 'human':
            pygame.event.pump()
            self.screen.fill((255, 255, 255))
            
            image_width = BG.get_width()
            self.screen.blit(BG, (self.bg_x_pos, 380))
            self.screen.blit(BG, (self.bg_x_pos + image_width, 380))
            self.bg_x_pos -= self.game_speed
            if self.bg_x_pos <= -image_width:
                self.bg_x_pos = 0
                
            for cloud in self.clouds:
                cloud.update(self.game_speed)
                cloud.draw(self.screen)

            self.player.draw(self.screen)
            for obs in self.obstacles:
                obs.draw(self.screen)

            pygame.display.update()


    
    def close(self):
        if pygame.get_init():
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = DinoEnv(render_mode="human")
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        env.clock.tick(30)
        done = terminated or truncated

    env.close()

