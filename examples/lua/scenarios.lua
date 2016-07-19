#!/usr/bin/env lua

----------------------------------------------------------------------
-- This script presents how to run some scenarios.
-- Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
-- <episodes> number of episodes are played.
-- Random combination of buttons is chosen for every action.
-- Game variables from state and last reward are printed.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require("vizdoom")

game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

game:loadConfig("../../examples/config/basic.cfg")
--game:loadConfig("../../examples/config/deadly_corridor.cfg")
--game:loadConfig("../../examples/config/deathmatch.cfg")
--game:loadConfig("../../examples/config/defend_the_center.cfg")
--game:loadConfig("../../examples/config/defend_the_line.cfg")
--game:loadConfig("../../examples/config/health_gathering.cfg")
--game:loadConfig("../../examples/config/my_way_home.cfg")
--game:loadConfig("../../examples/config/predict_position.cfg")
--game:loadConfig("../../examples/config/take_cover.cfg")

-- Makes the screen bigger to see more details.
game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
game:init()

-- Creates all possible actions depending on how many buttons there are.
actions_num = game:getAvailableButtonsSize()
actions = {}
for i = 1, actions_num do
    actions[i] = {}
    for j = 1, actions_num do
        actions[i][j] = 0
        if i == j then actions[i][j] = 1 end
    end
end


episodes = 10
sleep_time = 28

for i = 1, episodes do

    print("Episode #" .. i)

    -- Not needed for the first episode but the loop is nicer.
    game:newEpisode()
    while not game:isEpisodeFinished() do

        -- Gets the state and possibly to something with it
        state = game:getState()
        vars = state.gameVariables

        -- Makes a random action and save the reward.
        reward = game:makeAction(actions[math.random(1, actions_num)])

        print("State # " .. state.number)
        varsStr = "Game variables:"
        for k, a in pairs(vars) do varsStr = varsStr .. " " .. a end
        print(varsStr)
        print("Reward: " .. reward)
        print("=====================")

        -- Sleep some time because processing is too fast to watch.
        if sleep_time > 0 then
            vizdoom.sleep(sleep_time)
        end
    end

    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()

