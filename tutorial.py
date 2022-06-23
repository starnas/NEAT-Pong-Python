from locale import locale_alias
import pygame
from pong import Game
import neat
import os
import pickle


class PongGame:

    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.right_paddle = self.game.right_paddle
        self.left_paddle = self.game.left_paddle
        self.ball = self.game.ball

    def test_ai(self):
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(120)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                game.move_paddle(left=True, up=False)

            game_info = game.loop()
            print(game_info.left_score, game_info.right_score)

            game.loop()
            game.draw(True, True)
            pygame.display.update()

        pygame.quit()

# how good? dependent on opponent
# each ai vs every other ai


def eval_genomes(genomes, config):
    pass


def run_neat(config):

    # initialize population using the configuration file
    p = neat.Population(config)

    # restoring from a checkpoint
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-x')

    # report to std. output, fitness etc.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # save a checpoint after x generations - so you can restart
    p.add_reporter(neat.Checkpointer(1))

    # run n number of generations
    p.run(eval_genomes, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    # pass props from config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    run_neat(config)
