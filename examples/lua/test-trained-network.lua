-- Eugenio Culurciello
-- December 2016
-- test a trained deep Q learning neural network 

local base_path = "../../" -- path to ViZDoom's root dir

require "vizdoom"
require 'nn'
require 'image'

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {}
opt.fpath = arg[1]
if not opt.fpath then print('missing arg #1: th test.lua results/model-20.net') return end

-- load trained network:
local model = torch.load(opt.fpath)

local config_file_path = base_path.."scenarios/simpler_basic.cfg"
-- local config_file_path = base_path.."scenarios/rocket_basic.cfg"
-- local config_file_path = base_path.."scenarios/basic.cfg"

-- Doom game actions:
local actions = {
    [1] = torch.Tensor({1,0,0}),
    [2] = torch.Tensor({0,1,0}),
    [3] = torch.Tensor({0,0,1})
}

-- Other parameters
local resolution = {30, 45} -- Y, X sizes of rescaled state / game screen

-- Converts and down-samples the input image
local function preprocess(inImage)
  return image.scale(inImage, unpack(resolution))
end

-- Creates and initializes ViZDoom environment:
function initializeViZdoom(config_file_path)
    print("Initializing doom...")
    game = vizdoom.DoomGame()
    game:setViZDoomPath(base_path.."bin/vizdoom")
    game:setDoomGamePath(base_path.."scenarios/freedoom2.wad")
    game:loadConfig(config_file_path)
    game:setMode(vizdoom.Mode.PLAYER)
    game:setScreenFormat(vizdoom.ScreenFormat.GRAY8)
    game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
    game:init()
    print("Doom initialized.")
    return game
end

function getQValues(state)
    return model:forward(state)
end

function getBestAction(state)
    local q = getQValues(state:float():reshape(1, 1, resolution[1], resolution[2]))
    local max, index = torch.max(q, 1)
    local action = index[1]
    return action, q
end

-- Create Doom instance:
local game = initializeViZdoom(config_file_path)

-- Reinitialize the game with window visible:
game:setWindowVisible(true)
game:setMode(vizdoom.Mode.ASYNC_PLAYER)
game:init()

for i = 1, 20 do
    game:newEpisode()
    while not game:isEpisodeFinished() do
        local state = preprocess(game:getState().screenBuffer:float():div(255))
        local best_action_index = getBestAction(state)

        game:makeAction(actions[best_action_index])
        for j = 1, 12 do
            game:advanceAction()
        end
    end

    -- Sleep between episodes:
    sys.sleep(1)
    local score = game:getTotalReward()
    print("Total score: ", score)
end

game:close()
