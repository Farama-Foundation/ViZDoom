function vizdoom.DoomTicsToMs(tics, ticrate)
   return 1000.0 / ticrate * tics
end

function vizdoom.MsToDoomTics(ms, ticrate)
   return ticrate / 1000.0 * ms
end

function vizdoom.DoomFixedToDouble(doomFixed)
   return doomFixed / 65536.0
end

-- Missing: isBinaryButton, isDeltaButton
