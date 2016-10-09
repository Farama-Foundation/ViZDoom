function readFile(path)
   local file = io.open(path)
   local content = file:read("*a")
   file:close()
   return content
end

local ffi = require("ffi")

ffi.cdef("typedef struct DoomGame DoomGame;")
ffi.cdef(readFile(paths.thisfile("vizdoom.inl")))

local lib = ffi.load("../../bin/lua/libvizdoom.so")

vizdoom["lib"] = lib
