#!/usr/bin/python
from common import *
from vizia import *
from tqdm import tqdm

skiprate = 7
np.set_printoptions(precision=4, suppress = True)
savefile = None
#savefile = "params/basic_120to60"
savefile = "params/exp_skip"+str(skiprate)
#savefile = "params/s1b_120_to60_skip1"
loadfile = None

results_savefile = "results/res_exp_skip"+str(skiprate)

game = DoomGame()
game.load_config("config_common.properties")
#game.load_config("config_health_guided.properties")
game.load_config("config_basic.properties")


#game.clear_available_game_variables()
print "Initializing DOOM ..."
game.init()
print "\nDOOM initialized."


if loadfile:
    engine = QEngine.load(game, loadfile)

else:
    engine_args = engine_setup_basic(game)
    engine_args["skiprate"] = skiprate
    engine = QEngine(**engine_args)

if results_savefile:
    res_f = open(results_savefile, 'w')
    res_f.write(engine.params_to_print())
    res_f.write("Epoch | time | overall_time| steps | mean result| results\n")
print "\nNetwork architecture:"
for p in get_all_param_values(engine.get_network()):
	print p.shape



epochs = np.inf
training_steps_per_epoch = 5000
test_episodes_per_epoch = 100
test_frequency = 1;
overall_start = time()


epoch = 0
print "\nLearning ..."
while epoch < epochs:
    print "\nEpoch", epoch
    
    if training_steps_per_epoch>0:
        rewards = []
        start = time()

        engine.new_episode(True)
        for step in tqdm(range(training_steps_per_epoch)):
            if game.is_episode_finished():
                r = game.get_summary_reward()
                rewards.append(r)
                engine.new_episode(True)
            engine.make_learning_step()
        end = time()
        
        train_time = sec_to_str(end - start)
        print "Train:"
        print engine.get_actions_stats(True)
        mean_loss = engine._evaluator.get_mean_loss()
        print "steps:", engine.get_steps(), ", mean:", np.mean(rewards), ", max:", np.max(
            rewards), "min:", np.min(rewards), "mean_loss:",mean_loss, "eps:", engine.get_epsilon()
        print "t:", train_time
            
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
    
    overall_end = time()
    overall_time = sec_to_str(overall_end - overall_start)
    if results_savefile:
        res_f.write( str(epoch)+"| " + train_time + "| " +overall_time +"| "+ str(engine.get_steps()) + "| " +str(m) +"| ")
        for r in rewards:
            res_f.write(str(r) +" ")
        res_f.write("\n")

    epoch += 1

    print ""

    if savefile:
        engine.save(savefile)

    
    print "Elapsed time:", overall_time
    print "========================="


overall_end = time()
print "Elapsed time:", sec_to_str(overall_end - overall_start)
