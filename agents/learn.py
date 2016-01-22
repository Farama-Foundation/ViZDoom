#!/usr/bin/python
from common import *
from vizia import *
from tqdm import tqdm

np.set_printoptions(precision=4, suppress = True)
savefile = None
#savefile = "params/basic_120to60"
#savefile = "params/health_guided_60_skip8"
#savefile = "params/s1b_120_to60_skip1"
loadfile = None


game = DoomGame()
game.load_config("config_common.properties")
game.load_config("config_health_guided.properties")
#game.load_config("config_basic.properties")
print "Initializing DOOM ..."
game.init()
print "\nDOOM initialized."


if loadfile:
    engine = QEngine.load(game, loadfile)
    #engine.set_epsilon(0.0)
else:
    engine_args = engine_setup_basic(game)

    engine = QEngine(**engine_args)

print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
	print p.shape


epochs = np.inf
training_episodes_per_epoch = 400
test_episodes_per_epoch = 100
test_frequency = 1;
overall_start = time()


epoch = 0
print "\nLearning ..."
while epoch < epochs:
    engine.learning_mode = True
    rewards = []
    start = time()
    print "\nEpoch", epoch
    
    for episode in tqdm(range(training_episodes_per_epoch)):
        r = engine.run_episode()
        rewards.append(r)
        
    end = time()
    
    print "Train:"
    print engine.get_actions_stats(True)
    mean_loss = engine._evaluator.get_mean_loss()
    print "steps:", engine.get_steps(), ", mean:", np.mean(rewards), ", max:", np.max(
        rewards), "min:", np.min(rewards), "mean_loss:",mean_loss, "eps:", engine.get_epsilon()
    print "t:", sec_to_str(end - start)
        
    # learning mode off
    if (epoch+1) % test_frequency == 0 and test_episodes_per_epoch > 0:
        engine.learning_mode = False
        rewards = []

        start = time()
        for test_episode in tqdm(range(test_episodes_per_epoch)):
            r = engine.run_episode()
            rewards.append(r)
        end = time()
        
        print "Test"
        print engine.get_actions_stats(clear=True, norm=False)
        m = np.mean(rewards)
        print "steps:", engine.get_steps(), ", mean:", m, "max:", np.max(rewards), "min:", np.min(rewards)
        print "t:", sec_to_str(end - start)
    epoch += 1

    print ""

    if savefile:
        engine.save(savefile)

    overall_end = time()
    print "Elapsed time:", sec_to_str(overall_end - overall_start)
    print "========================="


overall_end = time()
print "Elapsed time:", sec_to_stri(overall_end - overall_start)
