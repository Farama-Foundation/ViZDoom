require 'sys'

local vizdoom = require("../../src/lib_lua/init.lua")

local Button = vizdoom.Button
local Mode = vizdoom.Mode
local GameVariable = vizdoom.GameVariable
local ScreenFormat = vizdoom.ScreenFormat
local ScreenResolution = vizdoom.ScreenResolution

-- Create DoomGame instance. It will run the game and communicate with you.
local game = vizdoom.ViZDoomLua()

-- Some configuration. Engine, scenario, maps
game:setViZDoomPath("../../bin/vizdoom");
game:setDoomGamePath("../../scenarios/freedoom2.wad")
game:setDoomScenarioPath("../../scenarios/basic.wad");
game:setDoomMap("map01");

-- Screen and rendering settings
game:setScreenResolution(ScreenResolution.RES_1280X1024);
game:setScreenFormat(ScreenFormat.RGB24);
game:setRenderHud(false);
game:setRenderCrosshair(false);
game:setRenderWeapon(true);
game:setRenderDecals(false);
game:setRenderParticles(false);

game:addAvailableButton(Button.MOVE_LEFT);
game:addAvailableButton(Button.MOVE_RIGHT);
game:addAvailableButton(Button.ATTACK);
game:addAvailableGameVariable(GameVariable.AMMO2);

game:setEpisodeTimeout(200);
game:setEpisodeStartTime(10);
game:setWindowVisible(true);
game:setSoundEnabled(true);

game:setLivingReward(-1.0)
game:setMode(Mode.PLAYER);
game:init();

-- Define actions
local actions = {
   [1] = torch.IntTensor({1,0,0}),
   [2] = torch.IntTensor({0,1,0}),
   [3] = torch.IntTensor({0,0,1})
}

-- Some experiment variables
local episodes = 10
local sleepTime = 0.05

-- Will be allocated and populated by getState
local state, reward

--------------------------------------------------------------------------------
-- Main game loop

for i=1, episodes do
   print("Episode #"..i)

   game:newEpisode()

   while not game:isEpisodeFinished() do

      -- get the GameState
      state = game:getState()

      -- Make a random action
      local action = actions[torch.random(#actions)]
      reward = game:makeAction(action)

      -- Print state's game variables.
      --print("State #"..state.number)
      --print("Game variables: ", state.gameVariables[1])
      --print("Reward: ", reward)

      if sleepTime > 0 then sys.sleep(sleepTime) end
   end
   -- Check how the episode went.
   print("Episode finished.")
   print("total reward:", game:getTotalReward())
   print("************************")
end

game:close()
