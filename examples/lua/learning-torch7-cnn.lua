#!/usr/bin/env th

-- E. Culurciello, December 2016
-- based on https://github.com/Marqt/ViZDoom/blob/master/examples/python/learning_tensorflow.py
-- use CNN and DQN to train simple Doom scenarios

-- "th learning-torch7-cnn.lua" to train a neural network from scratch, and test it
-- "th learning-torch7-cnn.lua --skipLearning" to only test a pre-trained neural network

local base_path = "../../" -- path to ViZDoom's root dir

require "vizdoom"
require 'nn'
require 'torch'
require 'sys'
require 'image'
require 'optim'
require 'xlua'

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --gridSize            (default 30)         default screen resized for neural net input
  --discount            (default 0.99)       discount factor in learning
  --epsilon             (default 1)          initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.1)        final value of ϵ-greedy action selection
  
  Training parameters:
  --skipLearning                             skip learning and just test
  --threads               (default 8)        number of threads used by BLAS routines
  --seed                  (default 1)        initial random seed
  -r,--learningRate       (default 0.00025)  learning rate
  --batchSize             (default 64)       batch size for training
  --maxMemory             (default 1e4)      Experience Replay buffer memory
  --epochs                (default 20)       number of training steps to perform

  -- Q-learning settings
  --learningStepsEpoch    (default 2000)     Learning steps per epoch
  --clampReward                              clamp reward to -1, 1

  -- Training regime
  --testEpisodesEpoch     (default 100)      test episodes per epoch
  --frameRepeat           (default 12)       repeat frame in test mode
  --episodesWatch         (default 10)       episodes to watch after training
  
  Display and save parameters:
  --display                                  display stuff
  --saveDir          (default './results')   subdirectory to save experiments in
  --load                  (default '')       load neural network to test
]]

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.saveDir)

-- Other parameters:
local resolution = {opt.gridSize, opt.gridSize*1.5} -- Y, X sizes of rescaled state / game screen
local colors = sys.COLORS

-- Configuration file path
local config_file_path = base_path.."scenarios/simpler_basic.cfg"
-- local config_file_path = base_path.."scenarios/rocket_basic.cfg"
-- local config_file_path = base_path.."scenarios/basic.cfg"

-- Doom basic scenario actions:
local actions = {
    [1] = torch.Tensor({1,0,0}),
    [2] = torch.Tensor({0,1,0}),
    [3] = torch.Tensor({0,0,1})
}

-- Converts and down-samples the input image:
local function preprocess(inImage)
    inImage = inImage:float():div(255)
  return image.scale(inImage, unpack(resolution))
end

-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    local channels = 1
    memory.s1 = torch.zeros(capacity, channels, resolution[1], resolution[2])
    memory.s2 = torch.zeros(capacity, channels, resolution[1], resolution[2])
    memory.a = torch.ones(capacity)
    memory.r = torch.zeros(capacity)
    memory.isterminal = torch.zeros(capacity)

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    -- batch buffers:
    memory.bs1 = torch.zeros(opt.batchSize, channels, resolution[1], resolution[2])
    memory.bs2 = torch.zeros(opt.batchSize, channels, resolution[1], resolution[2])
    memory.ba = torch.ones(opt.batchSize)
    memory.br = torch.zeros(opt.batchSize)
    memory.bisterminal = torch.zeros(opt.batchSize)

    function memory.addTransition(s1, action, s2, isterminal, reward)
        if memory.pos == 0 then memory.pos = 1 end -- tensors do not have 0 index items!
        memory.s1[{memory.pos, 1, {}, {}}] = s1
        memory.a[memory.pos] = action
        if not isterminal then
            memory.s2[{memory.pos, 1, {}, {}}] = s2
        end
        memory.isterminal[memory.pos] = isterminal and 1 or 0 -- boolean stored as 0 or 1 in memory!
        memory.r[memory.pos] = reward

        memory.pos = (memory.pos + 1) % memory.capacity
        memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getSample(sampleSize)
        for i=1,sampleSize do
            local ri = torch.random(1, memory.size-1)
            memory.bs1[i] = memory.s1[ri]
            memory.bs2[i] = memory.s2[ri]
            memory.ba[i] = memory.a[ri]
            memory.bisterminal[i] = memory.isterminal[ri]
            memory.br[i] = memory.r[ri]
        end
        return memory.bs1, memory.ba, memory.bs2, memory.bisterminal, memory.br
    end

end

local sgdParams = {
    learningRate = opt.learningRate,
}

local model, criterion
local function createNetwork(nAvailableActions)
    -- create CNN model:
    model = nn.Sequential()
    model:add(nn.SpatialConvolution(1,8,6,6,3,3))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(8,8,3,3,2,2))
    model:add(nn.ReLU())
    model:add(nn.View(8*4*6))
    model:add(nn.Linear(8*4*6, 128))
    model:add(nn.ReLU())
    model:add(nn.Linear(128, nAvailableActions))

    criterion = nn.MSECriterion()
end

local function learnBatch(s1, target_q)

    local params, gradParams = model:getParameters()
    
    local function feval(x_new)
        gradParams:zero()
        local predictions = model:forward(s1)
        local loss = criterion:forward(predictions, target_q)
        local gradOutput = criterion:backward(predictions, target_q)
        model:backward(s1, gradOutput)
        return loss, gradParams
    end

    local _, fs = optim.rmsprop(feval, params, sgdParams)
    return fs[1] -- loss
end

local function getQValues(state)
    return model:forward(state)
end

local function getBestAction(state)
    local q = getQValues(state:float():reshape(1, 1, resolution[1], resolution[2]))
    local max, index = torch.max(q, 1)
    local action = index[1]
    return action, q
end

local function learnFromMemory()
    -- Learns from a single transition (making use of replay memory)
    -- s2 is ignored if s2_isterminal

    -- Get a random minibatch from the replay memory and learns from it
    if memory.size > opt.batchSize then
        local s1, a, s2, isterminal, r = memory.getSample(opt.batchSize)
        if opt.clampReward then r = r:clamp(-1,1)  end -- clamping of reward!

        local q2 = torch.max(getQValues(s2), 2) -- get max q for each sample of batch
        local target_q = getQValues(s1):clone()
     
        -- target differs from q only for the selected action. The following means:
        -- target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        for i=1,opt.batchSize do
            target_q[i][a[i]] = r[i] + opt.discount * (1 - isterminal[i]) * q2[i]
        end
        learnBatch(s1, target_q)
    end
end

local function performLearningStep(epoch)
    -- Makes an action according to eps-greedy policy, observes the result
    -- (next state, reward) and learns from the transition

    local function explorationRate(epoch)
        --  Define exploration rate change over time:
        local start_eps = opt.epsilon
        local end_eps = opt.epsilonMinimumValue
        local const_eps_epochs = 0.1 * opt.epochs  -- 10% of learning time
        local eps_decay_epochs = 0.6 * opt.epochs  -- 60% of learning time

        if epoch < const_eps_epochs then
            return start_eps
        elseif epoch < eps_decay_epochs then
            -- Linear decay:
            return start_eps - (epoch - const_eps_epochs) /
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else
            return end_eps
        end
    end

    local s1 = preprocess(game:getState().screenBuffer)

    -- With probability eps make a random action:
    local eps = explorationRate(epoch)
    local a, s2
    if torch.uniform() <= eps then
        a = torch.random(1, #actions)
    else
        -- Choose the best action according to the network:
        a = getBestAction(s1)
    end
    local reward = game:makeAction(actions[a], opt.frameRepeat)

    local isterminal = game:isEpisodeFinished()
    if not isterminal then s2 = preprocess(game:getState().screenBuffer) else s2 = nil end

    -- Remember the transition that was just experienced:
    memory.addTransition(s1, a, s2, isterminal, reward)

    learnFromMemory()

    return eps
end

-- Creates and initializes ViZDoom environment:
local function initializeViZdoom(config_file_path)
    print("Initializing doom...")
    game = vizdoom.DoomGame()
    game:setViZDoomPath(base_path.."bin/vizdoom")
    game:setDoomGamePath(base_path.."scenarios/freedoom2.wad")
    game:loadConfig(config_file_path)
    game:setWindowVisible(opt.display)
    game:setMode(vizdoom.Mode.PLAYER)
    game:setScreenFormat(vizdoom.ScreenFormat.GRAY8)
    game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
    game:init()
    print("Doom initialized.")
    return game
end

local function main()
    -- Create Doom instance:
    local game = initializeViZdoom(config_file_path)

    -- Action = which buttons are pressed:
    local n = game:getAvailableButtonsSize()
    
    -- Create replay memory which will store the play data:
    ReplayMemory(opt.maxMemory)

    createNetwork(#actions)
    
    print("Starting the training!")

    local timeStart = sys.tic()
    if not opt.skipLearning then
        local epsilon
        for epoch = 1, opt.epochs do
            print(string.format(colors.green.."\nEpoch %d\n-------", epoch))
            local trainEpisodesFinished = 0
            local trainScores = {}

            print(colors.red.."Training...")
            game:newEpisode()
            for learning_step=1, opt.learningStepsEpoch do
                xlua.progress(learning_step, opt.learningStepsEpoch)
                epsilon = performLearningStep(epoch)
                if game:isEpisodeFinished() then
                    local score = game:getTotalReward()
                    table.insert(trainScores, score)
                    game:newEpisode()
                    trainEpisodesFinished = trainEpisodesFinished + 1
                end
            end

            print(string.format("%d training episodes played.", trainEpisodesFinished))

            trainScores = torch.Tensor(trainScores)

            print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f", 
                trainScores:mean(), trainScores:std(), trainScores:min(), trainScores:max()))
            -- print('Epsilon value', epsilon)

            print(colors.red.."\nTesting...")
            local testEpisode = {}
            local testScores = {}
            for testEpisode=1, opt.testEpisodesEpoch do
                xlua.progress(testEpisode, opt.testEpisodesEpoch)
                game:newEpisode()
                while not game:isEpisodeFinished() do
                    local state = preprocess(game:getState().screenBuffer)
                    local bestActionIndex = getBestAction(state)
                    
                    game:makeAction(actions[bestActionIndex], opt.frameRepeat)
                end
                local r = game:getTotalReward()
                table.insert(testScores, r)
            end

            testScores = torch.Tensor(testScores)
            print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f",
                testScores:mean(), testScores:std(), testScores:min(), testScores:max()))

            print("Saving the network weigths to:", opt.saveDir)
            torch.save(opt.saveDir..'/model-cnn-dqn-'..epoch..'.net', model:float():clearState())
            
            print(string.format(colors.cyan.."Total elapsed time: %.2f minutes", sys.toc()/60.0))
        end
    else
        if opt.load == '' then print('Missing neural net file to load!') os.exit() end
        model = torch.load(opt.load) -- otherwise load network to test!
    end
    
    game:close()
    print("======================================")
    print("Training finished. It's time to watch!")

    -- Reinitialize the game with window visible:
    game:setWindowVisible(true)
    game:setMode(vizdoom.Mode.ASYNC_PLAYER)
    game:init()

    for i = 1, opt.episodesWatch do
        game:newEpisode()
        while not game:isEpisodeFinished() do
            local state = preprocess(game:getState().screenBuffer)
            local bestActionIndex = getBestAction(state)

            game:makeAction(actions[bestActionIndex])
            for j = 1, opt.frameRepeat do
                game:advanceAction()
            end
        end

        -- Sleep between episodes:
        sys.sleep(1)
        local score = game:getTotalReward()
        print("Total score: ", score)
    end
    game:close()
end

-- run main program:
main()
