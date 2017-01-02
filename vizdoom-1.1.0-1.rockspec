package = "vizdoom"
version = "1.1.0-1"

source = {
    url = "git://github.com/Marqt/ViZDoom",
    tag = "1.1.0-luarocks"
}

description = {
    summary = "Reinforcement learning platform based on Doom",
    detailed = [[
        ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer).
        It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.
    ]],
    homepage = "http://vizdoom.cs.put.edu.pl/",
    --issues_url = "https://github.com/Marqt/ViZDoom/issues"
    --labels = {"vizdoom", "doom", "ai", "deep learning", "reinforcement learning", "research"}
}

supported_platforms = {"unix"}

dependencies = {
    "torch >= 7.0",
    "image >= 1.0",
}

build = {
    type = "command",
    build_command = [[
        rm -f CMakeCache.txt
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LUA=ON -DLUA_LIBRARIES="$(LUA_LIBDIR)/libluajit.so" -DLUA_INCLUDE_DIR="$(LUA_INCDIR)" && $(MAKE)
    ]],
    install_command = [[
        mkdir -p $(LUA_LIBDIR)/lua/5.1/vizdoom
        cp -r ./bin/lua/luarocks_package/* $(LUA_LIBDIR)/lua/5.1/vizdoom
        mkdir -p $(LUA_LIBDIR)/../share/lua/5.1/vizdoom
        cp -r ./bin/lua/luarocks_shared_package/* $(LUA_LIBDIR)/../share/lua/5.1/vizdoom
    ]]
}