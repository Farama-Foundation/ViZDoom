require("vizdoom.vizdoom")
local module = vizdoom
local torch = require("torch")
local ffi = require("ffi")

function vizdoom.DoomGame:getState()
    local _state = self:_getState()

    local width = self:getScreenWidth();
    local height = self:getScreenHeight();
    local channels = self:getScreenChannels();
    local size = channels * height * width;

    local state = {
        ["number"] = _state.number,
        ["game_variables"] = _state.number,

        ["screenBuffer"] = nil,
        ["depthBuffer"] = nil,
        ["labelsBuffer"] = nil,
        ["automapBuffer"] = nil,

        ["labels"] = _state.labels
    }

    state.gameVariables = torch.IntTensor(_state.gameVariables)

    local buffer = torch.ByteTensor(channels, height, width)
    ffi.copy(buffer:data(), _state.screenBuffer, size)
    state.screenBuffer = buffer;

    if _state.depthBuffer then
        local buffer = torch.ByteTensor(height, width)
        ffi.copy(buffer:data(), _state.depthBuffer, height * width)
        state.depthBuffer = buffer;
    end

    if _state.labelsBuffer then
        local buffer = torch.ByteTensor(height, width)
        ffi.copy(buffer:data(), _state.labelsBuffer, height * width)
        state.labelsBuffer = buffer;
    end

    if _state.automapBuffer then
        local buffer = torch.ByteTensor(channels, height, width)
        ffi.copy(buffer:data(), _state.automapBuffer, size)
        state.automapBuffer = buffer;
    end

    return state
end

function vizdoom.DoomGame:setAction(action)
    if torch.type(action) ~= "table" then
        action = torch.totable(action);
    end

    self:_setAction(torch.totable(action))
end


function vizdoom.DoomGame:makeAction(action, tics)
    tics = tics or 1

    if torch.type(action) ~= "table" then
        action = torch.totable(action);
    end

    return self:_makeAction(action, tics)
end

return module