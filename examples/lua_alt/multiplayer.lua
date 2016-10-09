local threads = require 'threads'

local noOfPlayers = 2
local episodes = 10

local pool = threads.Threads(

   -- spawned threads
   noOfPlayers,

   -- Set-up the environment on each thread
   function()
      -- for some reason 'require' has an unexpected behavior when called
      -- from 'threads' to execute a module outside this director.
      vizdoom = dofile("../../src/lib_lua/init.lua")
      GameVariable = vizdoom.GameVariable

      actions = {
         [1] = torch.IntTensor({1,0,0}),
         [2] = torch.IntTensor({0,1,0}),
         [3] = torch.IntTensor({0,0,1})
      }
   end,

   function()

      function player1()
         -- Create DoomGame instance.
         game = vizdoom.ViZDoomLua()

         -- Config
         game:loadConfig("../config/multi_duel.cfg")
         game:addGameArgs("-host 2 -deathmatch "..
                          "+timelimit 1.0 +sv_spawnfarthest 1")
         game:addGameArgs("+name Player1 +colorset 0")
         game:init()

         for i=1, episodes do

            while not game:isEpisodeFinished() do
               if game:isPlayerDead() then
                  game:respawnPlayer()
               end
               -- Make a random action
               local action = actions[torch.random(#actions)]
               reward = game:makeAction(action)
            end
           print("Player1 frags:", game:getGameVariable(GameVariable.FRAGCOUNT))
           game:newEpisode()
         end
         game:close()
      end

      function player2()
         -- Create DoomGame instance.
         game = vizdoom.ViZDoomLua()

         -- Config
         game:loadConfig("../config/multi_duel.cfg")
         game:addGameArgs("-join 127.0.0.1")
         game:addGameArgs("+name Player2 +colorset 3")
         game:init()

         for i=1, episodes do

            while not game:isEpisodeFinished() do
               if game:isPlayerDead() then
                  game:respawnPlayer()
               end
               -- Make a random action
               local action = actions[torch.random(#actions)]
               reward = game:makeAction(action)
            end
           print("Player1 frags:", game:getGameVariable(GameVariable.FRAGCOUNT))
           game:newEpisode()
         end
         game:close()
      end
   end
)

pool:specific(true)

pool:addjob(
   1,
   function()
      player1()
   end
)
pool:addjob(
   2,
   function()
      player2()
   end
)

pool:synchronize()
pool:terminate()
print("Done")
