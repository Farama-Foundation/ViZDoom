#!/usr/bin/python
from common import *

savefile = None

#savefile = "params/basic_120to60"
savefile = "params/health_guided_120_to60_skip8_3l_f48"
#savefile = "params/center_120_to80_skip4"
#savefile = "params/s1b_120_to60_skip1"
loadfile = None


game = setup_vizia(scenario=health_guided, init=True)

engine = create_engine(game)

if loadfile:
    engine.load_params(loadfile)

print "\nCreated network params:"
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
    
    for episode in range(training_episodes_per_epoch):
        r = engine.run_episode()
        rewards.append(r)
        
    end = time()
    
    print "Train:"
    print engine.get_actions_stats(True)
    mean_loss = engine._evaluator.get_mean_loss()
    print "steps:", engine.get_steps(), ", mean:", np.mean(rewards), ", max:", np.max(
        rewards),"mean_loss:",mean_loss, "eps:", engine.get_epsilon()
    print "t:", round(end - start, 2)
        
    # learning mode off
    if (epoch+1) % test_frequency == 0 and test_episodes_per_epoch > 0:
        engine.learning_mode = False
        rewards = []

        start = time()
        for test_episode in range(test_episodes_per_epoch):
            r = engine.run_episode()
            rewards.append(r)
        end = time()
        
        print "Test"
        print engine.get_actions_stats(clear=True, norm=False)
        m = np.mean(rewards)
        print "steps:", engine.get_steps(), ", mean:", m, "max:", np.max(rewards)
        print "t:", round(end - start, 2)
    epoch += 1

    print ""

    if savefile:
        engine.save_params(savefile)

    overall_end = time()
    print "Elapsed time:", round(overall_end - overall_start,2), "s"  
    print "========================="


overall_end = time()
print "Elapsed time:", round(overall_end - overall_start,2), "s"  
