local vizdoom = require("../../src/lib_lua/init.lua")

local DoomGame = vizdoom.ViZDoomLua
local ScreenResolution = vizdoom.ScreenResolution
local ScreenFormat = vizdoom.ScreenFormat
local Button = vizdoom.Button
local GameVariable = vizdoom.GameVariable
local Mode = vizdoom.Mode

-- Create DoomGame instance. It will run the game and communicate with you.
local game = DoomGame()


game:loadConfig("../../examples/config/deathmatch.cfg")
game:setViZDoomPath("../../bin/vizdoom")

game:addGameArgs("+freelook 1")

game:setScreenResolution(ScreenResolution.RES_640X480)

game:setWindowVisible(true)

-- Enables spectator mode, so You can play and agent watch your actions.
-- You can only use the buttons selected as available.
game:setMode(Mode.SPECTATOR)

game:init()


local episodes = 10

for i = 1, 10 do
   io.write("Episode #" .. i .. "\n")

   game:newEpisode()

   while not game:isEpisodeFinished() do
      -- Get the state.
      local s = game:getState()

      -- Advances action - lets You play next game tic.
      game:advanceAction()

      -- You can also advance a few tics at once.
      -- game->advanceAction(4);

      -- Get the last action performed by You.
      local a = game:getLastAction();

      -- And reward You get.
      local r = game:getLastReward();

      io.write("State #" .. s.number .. "\n")

      io.write("Game variables:")
      local v = s.gameVariables
      for i = 1, v:nElement() do io.write(" " .. v[i]) end
      io.write("\n")

      io.write("Action made: ")

      for i = 1, a:nElement() do io.write(" " .. a[i]) end
      io.write("\n")
      io.write("Action reward: " .. r .. "\n");
      io.write("=====================\n")

   end

   io.write("Episode finished.\n")
   io.write("Total reward: " .. game:getTotalReward() .. "\n")
   io.write("************************\n")

end

-- It will be done automatically in destructor but after close You can
-- init it again with different settings.
game:close()
