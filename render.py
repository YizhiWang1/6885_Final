import pygame
import random
from environment import EmotionWorld

def render():
    # create an EmotionWorld object
    world = EmotionWorld()

    # initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((world.width, world.height))  # create a window
    pygame.display.set_caption("Emotion World Simulation")  # set title

    clock = pygame.time.Clock()
    running = True

    while running:
        # handle events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # randomly trigger emotion propagation in each frame
        if random.random() < 0.5:  # increase the trigger probability
            emotion_type = random.choice(['joy', 'sadness', 'anger', 'fear', 'disgust'])
            target_pos = (random.randint(0, world.width), random.randint(0, world.height))
            affected_count = world._spread_emotion(emotion_type, target_pos)
            print(f"Triggered {emotion_type} at {target_pos}, affected {affected_count} people")


        # update the world state
        world._update_positions()
        world._update_emotions()
        chain_affected = world._spread_emotion_chain()
        print(f"Chain affected count: {chain_affected}")

    # render the current state of the world
        screen.fill((255, 255, 255))  # white background
        for person in world.people:
            # choose color based on emotion
            if person['emotion'] == 'joy':
                color = (0, 255, 0)  # green
            elif person['emotion'] == 'sadness':
                color = (0, 0, 255)  # blue
            elif person['emotion'] == 'anger':
                color = (255, 0, 0)  # red
            elif person['emotion'] == 'fear':
                color = (128, 0, 128)  # purple
            elif person['emotion'] == 'disgust':
                color = (255, 128, 0)  # orange
            else:
                color = (200, 200, 200)  # neutral emotion is gray

            position = person['position'].astype(int)
            pygame.draw.circle(screen, color, position, 5)

        pygame.display.flip()  # refresh the screen

        clock.tick(30)  # 30 frames per second

    pygame.quit()  # exit pygame

if __name__ == "__main__":
    render()
