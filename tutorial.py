from locale import locale_alias
import pygame
from pong import Game
import neat
import os
import pickle

# the game of Pong


class PongGame:

    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.right_paddle = self.game.right_paddle
        self.left_paddle = self.game.left_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):

        # create network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:

            # run at 60fps
            clock.tick(60)

            # close on quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            # keybindins for player - operating left paddle
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # have the neural net decide on output
            output = net.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))

            # get the highest number from output 1
            # 0 stay still
            # 1 move up
            # 2 move down
            decision = output.index(max(output))

            # based on the index, make a move
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            # start the game
            game_info = self.game.loop()

            # display
            #print(game_info.left_score, game_info.right_score)
            self.game.draw(draw_score=True, draw_hits=False)
            pygame.display.update()

        pygame.quit()

    # have the genomes play the game
    def train_ai(self, genome1, genome2, config):

        # create neural network for genomes, pass inputs, get outputs
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        # start and quit the game
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # get outputs from the nerual networks
            # numeric values for the neural networks
            output1 = net1.activate(
                (self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            output2 = net2.activate(
                (self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            # print(output1, output2)

            # get the highest number from output 1
            # 0 stay still
            # 1 move up
            # 2 move down
            decision1 = output1.index(max(output1))
            decision2 = output2.index(max(output2))

            # based on the index, make a move
            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            # get game info
            game_info = self.game.loop()

            # draw and update
            # self.game.draw(draw_score=False, draw_hits=True)
            # pygame.display.update()

            # stop after the first miss
            # calulates fitness
            # also stop after 50 hits
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calulcate_fitness(genome1, genome2, game_info)
                break

    def calulcate_fitness(self, genome1, genome2, game_info):

        # fitness is the number of hits
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

# how good? dependent on opponent
# each ai vs every other ai


def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    # genomes - list of tuples, genomeID and genome object
    # if every genome vs other genome then enumerate
    for i, (genome_id1, genome1) in enumerate(genomes):

        # if the lest index, break
        if i == len(genomes) - 1:
            break

        # set default fitness for a genome
        genome1.fitness = 0

        # riun each genome 1 vs other genomes exactly 1
        # fitness is the SUM of allgames
        for genome_id2, genome2 in genomes[i+1:]:

            # check if genome2 has a fitness value, not to overwrite it
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness

            # initiate the game
            game = PongGame(window, width, height)

            # train the ai
            game.train_ai(genome1, genome2, config)


def run_neat(config):

    # initialize population using the configuration file
    p = neat.Population(config)

    # restoring from a checkpoint
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-x')

    # report to std. output, fitness etc.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # save a checpoint after x generations - so you can restart
    p.add_reporter(neat.Checkpointer(1))

    # run n number of generations
    winner = p.run(eval_genomes, 50)

    # save the winner genome
    # saving the python object
    with open('best.pickle', 'wb') as f:
        pickle.dump(winner, f)


def test_ai(config):

    # define a game window
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    # load the best genome
    with open('best.pickle', 'rb') as f:
        winner = pickle.load(f)

    # start a game with the top ai
    game = PongGame(window, width, height)
    game.test_ai(winner, config)


# main loop
if __name__ == "__main__":

    # read config from local dir
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    # pass props from config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # run neat to get best ai
    # run_neat(config)

    # test yourself with the top ai
    test_ai(config)
