#!/usr/bin/env lua

----------------------------------------------------------------------
-- This script presents SPECTATOR mode. In SPECTATOR mode you play and
-- your agent can learn from it.
-- Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require("vizdoom")

game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

--game:loadConfig("../../examples/config/basic.cfg")
game:loadConfig("../../examples/config/deadly_corridor.cfg")
--game:loadConfig("../../examples/config/deathmatch.cfg")
--game:loadConfig("../../examples/config/defend_the_center.cfg")
--game:loadConfig("../../examples/config/defend_the_line.cfg")
--game:loadConfig("../../examples/config/health_gathering.cfg")
--game:loadConfig("../../examples/config/my_way_home.cfg")
--game:loadConfig("../../examples/config/predict_position.cfg")
--game:loadConfig("../../examples/config/take_cover.cfg")

-- Enables freelook in engine
game:addGameArgs("+freelook 1")

game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
game:setWindowVisible(true)

-- Enables labeling of the in game objects.
game:setLabelsBufferEnabled(true)

game:init()

episodes = 10

for i = 1, episodes do

    print("Episode #" .. i)

    game:newEpisode()

    while not game:isEpisodeFinished() do

        -- Gets the state.
        state = game:getState()

        -- Get labels buffer and labels data.
        labelsBuf = state.labelsBuffer
        labels = state.labels

        game:advanceAction()
        reward = game:getLastReward()

        print("State #" .. state.number)

        print("Labels:")

        -- Print information about objects visible on the screen.
        -- object_id identifies specific in game object.
        -- object_name contains name of object.
        -- value tells which value represents object in labels_buffer.
        for k, l in pairs(labels) do
            print("#" .. k .. ": object id: " .. l.objectId .. " object name: " .. l.objectName .. " label: " .. l.value)
        end

        print("Reward: " .. reward)
        print("=====================")

    end

    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()
