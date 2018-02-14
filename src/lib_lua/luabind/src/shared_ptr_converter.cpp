#define LUABIND_BUILDING
#include <luabind/shared_ptr_converter.hpp>

#include <luabind/lua_include.hpp>

static char state_unreferenced_callback_tag = 0;
LUABIND_API char luabind::detail::state_use_count_tag = 0;

void luabind::detail::shared_ptr_deleter::alter_use_count(
    lua_State* L, lua_Integer diff)
{
    lua_rawgetp(L, LUA_REGISTRYINDEX, &state_use_count_tag);
    lua_Integer uc = lua_tointeger(L, -1) + diff;
    lua_pop(L, 1);
    assert(uc >= 0);
    lua_pushinteger(L, uc);
    lua_rawsetp(L, LUA_REGISTRYINDEX, &state_use_count_tag);

    if (!uc)
    {
        if (state_unreferenced_fun cb = get_state_unreferenced_callback(L))
            cb(L);
    }
}

void luabind::set_state_unreferenced_callback(
    lua_State* L, state_unreferenced_fun cb)
{
    lua_pushcfunction(L, reinterpret_cast<lua_CFunction>(cb));
    lua_rawsetp(L, LUA_REGISTRYINDEX, &state_unreferenced_callback_tag);
}

luabind::state_unreferenced_fun
luabind::get_state_unreferenced_callback(
    lua_State* L)
{
    lua_rawgetp(L, LUA_REGISTRYINDEX, &state_unreferenced_callback_tag);
    state_unreferenced_fun cb = reinterpret_cast<state_unreferenced_fun>(
        lua_tocfunction(L, -1));
    lua_pop(L, 1);
    return cb;
}

bool luabind::is_state_unreferenced(lua_State* L)
{
    lua_rawgetp(L, LUA_REGISTRYINDEX, &detail::state_use_count_tag);
    lua_Integer uc = lua_tointeger(L, -1);
    lua_pop(L, 1);
    assert(uc >= 0);
    return uc <= 0;
}
