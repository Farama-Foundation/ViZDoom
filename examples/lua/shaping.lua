#!/usr/bin/env th

require "vizdoom"
require "torch"

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.DoomGame()

game:loadConfig("../../scenarios/health_gathering.cfg")
game:setViZDoomPath("../../bin/vizdoom")

game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)

game:init()

-- Define actions
local actions = {
    [1] = torch.IntTensor({1,0,0}),
    [2] = torch.IntTensor({0,1,0}),
    [3] = torch.IntTensor({0,0,1})
}

lastTotalShapingReward = 0

local episodes = 10

for i = 1, 10 do
   print("Episode #" .. i .. "\n")

   game:newEpisode()

   while not game:isEpisodeFinished() do
      -- Get the state.
      local state = game:getState()

      local action = actions[torch.random(#actions)]
      local reward = game:makeAction(action)


      local _ssr = game:getGameVariable(vizdoom.GameVariable.USER1)
      local ssr = vizdoom.doomFixedToNumber(_ssr)
      local sr = ssr - lastTotalShapingReward
      lastTotalShapingReward = ssr;

      print("State #" .. state.number)

      print("Healt: ", state.gameVariables[1])
      print("Action reward: ", reward);
      print("Action shaping reward: ", sr);
      print("=====================")

   end

   print("Episode finished.\n")
   print("Total reward: " .. game:getTotalReward() .. "\n")
   print("************************\n")

end

-- It will be done automatically in destructor but after close You can
-- init it again with different settings.
game:close()
