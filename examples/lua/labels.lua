#!/usr/bin/env th

----------------------------------------------------------------------
-- This script presents SPECTATOR mode. In SPECTATOR mode you play and
-- your agent can learn from it.
-- Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
--
-- To see the scenario description go to "../../scenarios/README.md"
----------------------------------------------------------------------

require "vizdoom"
require "torch"
require "image"

local game = vizdoom.DoomGame()

-- Choose scenario config file you wish to watch.
-- Don't load two configs cause the second will overrite the first one.
-- Multiple config files are ok but combining these ones doesn't make much sense.

game:loadConfig("../../scenarios/deadly_corridor.cfg")

game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
game:setWindowVisible(true)

-- Enables labeling of the in game objects.
game:setLabelsBufferEnabled(true)

game:clearAvailableGameVariables()
game:addAvailableGameVariable(vizdoom.GameVariable.POSITION_X)
game:addAvailableGameVariable(vizdoom.GameVariable.POSITION_Y)
game:addAvailableGameVariable(vizdoom.GameVariable.POSITION_Z)

game:init()

local actions = {
    [1] = torch.IntTensor({1,0,0}),
    [2] = torch.IntTensor({0,1,0}),
    [3] = torch.IntTensor({0,0,1})
}

local episodes = 10
local sleepTime = 0.028

for i = 1, episodes do

    print("Episode #" .. i)

    game:newEpisode()

    while not game:isEpisodeFinished() do

        -- Gets the state.
        local state = game:getState()

        -- Get labels buffer and labels data.
        local labelsBuf = state.labelsBuffer
        local labels = state.labels

        --image.display(labelsBuf)

        game:makeAction(actions[torch.random(#actions)])

        print("State #" .. state.number)
        print("Player position X: " .. state.gameVariables[1] .. " Y: " .. state.gameVariables[2] .. " Z: " .. state.gameVariables[3])
        print("Labels:")

        -- Print information about objects visible on the screen.
        -- object_id identifies specific in game object.
        -- object_name contains name of object.
        -- value tells which value represents object in labels_buffer.
        for k, l in pairs(labels) do
            print("Object id: " .. l.objectId .. " object name: " .. l.objectName .. " label: " .. l.value)

            print("Object position x: " .. l.objectPositionX .. " y: " .. l.objectPositionY .. " z: " .. l.objectPositionZ)

            -- Other available fields:
            --print("Object rotation angle " ..  l.object_angle .. " pitch: " ..  l.object_pitch .. " roll: " ..  l.object_roll)
            --print("Object velocity x: " ..  l.object_velocity_x .. " y: " ..  l.object_velocity_y .. " z: " ..  l.object_velocity_z)
            --print("Bounding box: x: " .. l.x .. " y: " .. l.y .. " width: " .. l.width .. " height: " .. l.height)
        end

        print("=====================")

        if sleepTime > 0 then
            sys.sleep(sleepTime)
        end

    end

    print("Episode finished.")
    print("total reward: " .. game:getTotalReward())
    print("************************")

end

game:close()
