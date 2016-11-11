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
    local shape = {}
    local format = self:getScreenFormat();

    if format == vizdoom.ScreenFormat.CRCGCB or format == vizdoom.ScreenFormat.CBCGCR then
        shape[1] = channels
        shape[2] = height
        shape[3] = width
    else
        shape[1] = height
        shape[2] = width
        shape[3] = channels
    end

    local state = {
        ["number"] = _state.number,
        ["game_variables"] = nil,

        ["screenBuffer"] = nil,
        ["depthBuffer"] = nil,
        ["labelsBuffer"] = nil,
        ["automapBuffer"] = nil,

        ["labels"] = _state.labels
    }

    if _state.gameVariables then
        state.gameVariables = torch.DoubleTensor(_state.gameVariables)
    end

    if channels > 1 then
        state.screenBuffer = torch.ByteTensor(shape[1], shape[2], shape[3])
    else
        state.screenBuffer = torch.ByteTensor(shape[1], shape[2])
    end
    ffi.copy(state.screenBuffer:data(), _state.screenBuffer, size)

    if _state.depthBuffer then
        state.depthBuffer = torch.ByteTensor(height, width)
        ffi.copy(state.depthBuffer:data(), _state.depthBuffer, height * width)
    end

    if _state.labelsBuffer then
        state.labelsBuffer = torch.ByteTensor(height, width)
        ffi.copy(state.labelsBuffer:data(), _state.labelsBuffer, height * width)
    end

    if _state.automapBuffer then
        if channels > 1 then
            state.automapBuffer = torch.ByteTensor(shape[1], shape[2], shape[3])
        else
            state.automapBuffer = torch.ByteTensor(shape[1], shape[2])
        end
        ffi.copy(state.automapBuffer:data(), _state.automapBuffer, size)
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

function vizdoom.DoomGame:getLastAction()
    return torch.IntTensor(self:_getLastAction())
end

return module