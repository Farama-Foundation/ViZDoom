local vizdoom = require("../../src/lib_lua/init.lua")

local DoomGame = vizdoom.ViZDoomLua
local ScreenResolution = vizdoom.ScreenResolution
local ScreenFormat = vizdoom.ScreenFormat
local Button = vizdoom.Button
local GameVariable = vizdoom.GameVariable
local Mode = vizdoom.Mode


io.write("\n\nSHAPING EXAMPLE\n\n")

-- Create DoomGame instance. It will run the game and communicate with you.
local game = DoomGame()


game:loadConfig("../../examples/config/health_gathering.cfg")
game:setViZDoomPath("../../bin/vizdoom")

game:setScreenResolution(ScreenResolution.RES_640X480)


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
   io.write("Episode #" .. i .. "\n")

   game:newEpisode()

   while not game:isEpisodeFinished() do
      -- Get the state.
      local s = game:getState()

      local action = actions[torch.random(#actions)]
      local reward = game:makeAction(action)


      local _ssr = game:getGameVariable(GameVariable.USER1)

      local ssr = vizdoom.DoomFixedToDouble(_ssr)

      local sr = ssr - lastTotalShapingReward

      lastTotalShapingReward = ssr;

      io.write("State #" .. s.number .. "\n")

      io.write("Healt: " .. s.gameVariables[1] .. "\n")
      io.write("Action reward: " .. reward .. "\n");
      io.write("Action shaping reward: " .. sr .. "\n");
      io.write("=====================\n")

   end

   io.write("Episode finished.\n")
   io.write("Total reward: " .. game:getTotalReward() .. "\n")
   io.write("************************\n")

end

-- It will be done automatically in destructor but after close You can
-- init it again with different settings.
game:close()
