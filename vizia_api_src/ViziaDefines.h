#ifndef __VIZIA_DEFINES_H__
#define __VIZIA_DEFINES_H__

#include <exception>
#include <SDL2/SDL.h>

namespace Vizia{

    class Exception : public std::exception {
    public:
        virtual const char* what() const throw(){ return "Unknown exception."; }
    };

    class SharedMemoryException : public Exception {
    public:
        const char* what() const throw(){ return "Shared memory error."; }
    };

    class MessageQueueException : public Exception {
    public:
        const char* what() const throw(){ return "Message queue error."; }
    };

    class DoomErrorException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViziaZDoom instance raported error."; }
    };

    class DoomUnexpectedExitException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViziaZDoom instance exited unexpectedly."; }
    };

    class DoomIsNotRunningException : public Exception {
    public:
        const char* what() const throw(){ return "Controlled ViziaZDoom instance is not running or not ready."; }
    };

    enum GameMode {
        PLAYER,
        SPECATOR
    };

    enum ScreenFormat {
        CRCGCB = 0,
        CRCGCBCA = 1,
        RGB24 = 2,
        RGBA32 = 3,
        ARGB32 = 4,
        CBCGCR = 5,
        CBCGCRCA = 6,
        BGR24 = 7,
        BGRA32 = 8,
        ABGR32 = 9,
        GRAY8 = 10,
        DOOM_256_COLORS = 11,
    };

    enum GameVar {
        KILLCOUNT = 0,
        ITEMCOUNT = 1,
        SECRETCOUNT = 2,
        HEALTH = 3,
        ARMOR = 4,
        SELECTED_WEAPON = 5,
        SELECTED_WEAPON_AMMO = 6,
        AMMO1 = 7,
        AMMO2 = 8,
        AMMO3 = 9,
        AMMO4 = 10,
        WEAPON1 = 11,
        WEAPON2 = 12,
        WEAPON3 = 13,
        WEAPON4 = 14,
        WEAPON5 = 15,
        WEAPON6 = 16,
        WEAPON7 = 17,
        KEY1 = 18,
        KEY2 = 19,
        KEY3 = 20,
        USER1,
        USER2,
        USER3,
        USER4,
        USER5,
        USER6,
        USER7,
        USER8,
        USER9,
        USER10,
        USER11,
        USER12,
        USER13,
        USER14,
        USER15,
        USER16,
        USER17,
        USER18,
        USER19,
        USER20,
        USER21,
        USER22,
        USER23,
        USER24,
        USER25,
        USER26,
        USER27,
        USER28,
        USER29,
        USER30,
        UNDEFINED_VAR
    };

    static const int UserVarsNumber = 30;

    enum Button {
        ATTACK = 0,
        USE = 1,
        JUMP = 2,
        CROUCH = 3,
        TURN180 = 4,
        ALTATTACK = 5,
        RELOAD = 6,
        ZOOM = 7,

        SPEED = 8,
        STRAFE = 9,

        MOVE_RIGHT = 10,
        MOVE_LEFT = 11,
        MOVE_BACK = 12,
        MOVE_FORWARD = 13,
        TURN_RIGHT = 14,
        TURN_LEFT = 15,
        LOOK_UP = 16,
        LOOK_DOWN = 17,
        MOVE_UP = 18,
        MOVE_DOWN = 19,
        //SHOWSCORES 20

        SELECT_WEAPON1 = 21,
        SELECT_WEAPON2 = 22,
        SELECT_WEAPON3 = 23,
        SELECT_WEAPON4 = 24,
        SELECT_WEAPON5 = 25,
        SELECT_WEAPON6 = 26,
        SELECT_WEAPON7 = 27,

        SELECT_NEXT_WEAPON = 28,
        SELECT_PREV_WEAPON = 29,
        UNDEFINED_BUTTON
    };

    static const int ButtonsNumber = 30;

    static const int DOOM_AMMO_CLIP = 0;
    static const int DOOM_AMMO_SHELL = 1;
    static const int DOOM_AMMO_ROCKET = 2;
    static const int DOOM_AMMO_CELL = 3;

    static const int DOOM_WEAPON_FIST = 0;
    static const int DOOM_WEAPON_CHAINSAW = 0;
    static const int DOOM_WEAPON_PISTOL = 1;
    static const int DOOM_WEAPON_SHOTGUN = 3;
    static const int DOOM_WEAPON_SSG = 3;
    static const int DOOM_WEAPON_SUPER_SHOTGUN = 3;
    static const int DOOM_WEAPON_CHAINGUN = 4;
    static const int DOOM_WEAPON_ROCKET_LUNCHER = 5;
    static const int DOOM_WEAPON_PLASMA_GUN = 6;
    static const int DOOM_WEAPON_BFG = 7;

    static const int DOOM_KEY_BLUE = 0;
    static const int DOOM_KEY_RED = 1;
    static const int DOOM_KEY_YELLOW = 2;

#define VK_2_SDLK(vk) vk = SDLK_ ## vk
#define VK_T_2_SDLK(vk, sk) vk = SDLK_ ## sk

    enum Key {
        VK_2_SDLK(TAB),
        VK_T_2_SDLK(CAPS_LOCK, CAPSLOCK),
        VK_2_SDLK(RSHIFT),
        VK_2_SDLK(RCTRL),
        VK_2_SDLK(RALT),
        VK_2_SDLK(LSHIFT),
        VK_2_SDLK(LCTRL),
        VK_2_SDLK(LALT),
        VK_2_SDLK(SPACE),
        VK_T_2_SDLK(ENTER, RETURN),
        VK_2_SDLK(BACKSPACE),
        VK_2_SDLK(ESCAPE),
        VK_T_2_SDLK(RIGHT_ARROW, RIGHT),
        VK_T_2_SDLK(LEFT_ARROW, LEFT),
        VK_T_2_SDLK(UP_ARROW, UP),
        VK_T_2_SDLK(DOWN_ARROW, DOWN),
        VK_2_SDLK(INSERT),
        VK_2_SDLK(END),
        VK_2_SDLK(HOME),
        VK_T_2_SDLK(PAGE_UP, PAGEUP),
        VK_T_2_SDLK(PAGE_DOWN, PAGEDOWN),
        VK_2_SDLK(F1),
        VK_2_SDLK(F2),
        VK_2_SDLK(F3),
        VK_2_SDLK(F4),
        VK_2_SDLK(F5),
        VK_2_SDLK(F6),
        VK_2_SDLK(F7),
        VK_2_SDLK(F8),
        VK_2_SDLK(F9),
        VK_2_SDLK(F10),
        VK_2_SDLK(F11),
        VK_2_SDLK(F12),

        VK_T_2_SDLK(CHAR_Q, q),
        VK_T_2_SDLK(CHAR_W, w),
        VK_T_2_SDLK(CHAR_E, e),
        VK_T_2_SDLK(CHAR_R, r),
        VK_T_2_SDLK(CHAR_T, t),
        VK_T_2_SDLK(CHAR_Y, y),
        VK_T_2_SDLK(CHAR_U, u),
        VK_T_2_SDLK(CHAR_I, i),
        VK_T_2_SDLK(CHAR_O, o),
        VK_T_2_SDLK(CHAR_P, p),
        VK_T_2_SDLK(CHAR_A, a),
        VK_T_2_SDLK(CHAR_S, s),
        VK_T_2_SDLK(CHAR_D, d),
        VK_T_2_SDLK(CHAR_F, f),
        VK_T_2_SDLK(CHAR_G, g),
        VK_T_2_SDLK(CHAR_H, h),
        VK_T_2_SDLK(CHAR_J, j),
        VK_T_2_SDLK(CHAR_K, k),
        VK_T_2_SDLK(CHAR_L, l),
        VK_T_2_SDLK(CHAR_Z, z),
        VK_T_2_SDLK(CHAR_X, x),
        VK_T_2_SDLK(CHAR_C, c),
        VK_T_2_SDLK(CHAR_V, v),
        VK_T_2_SDLK(CHAR_B, b),
        VK_T_2_SDLK(CHAR_N, n),
        VK_T_2_SDLK(CHAR_M, m),

        VK_T_2_SDLK(CHAR_1, 1),
        VK_T_2_SDLK(CHAR_2, 2),
        VK_T_2_SDLK(CHAR_3, 3),
        VK_T_2_SDLK(CHAR_4, 4),
        VK_T_2_SDLK(CHAR_5, 5),
        VK_T_2_SDLK(CHAR_6, 6),
        VK_T_2_SDLK(CHAR_7, 7),
        VK_T_2_SDLK(CHAR_8, 8),
        VK_T_2_SDLK(CHAR_9, 9),
        VK_T_2_SDLK(CHAR_0, 0),
        VK_T_2_SDLK(CHAR_MINUS, MINUS),
        VK_T_2_SDLK(CHAR_EQUALS, EQUALS),

        VK_T_2_SDLK(CHAR_COMMA, COMMA),
        VK_T_2_SDLK(CHAR_PERIOD, PERIOD),
        VK_T_2_SDLK(CHAR_LEFT_BRACKET, LEFTBRACKET),
        VK_T_2_SDLK(CHAR_RIGHT_BRACKET, RIGHTBRACKET),
        VK_T_2_SDLK(CHAR_SLASH, SLASH),
        VK_T_2_SDLK(CHAR_BACKSLASH, BACKSLASH),
        VK_T_2_SDLK(CHAR_SEMICOLON, SEMICOLON),
        VK_T_2_SDLK(CHAR_QUOTE, QUOTE),

        UNDEFINED_KEY
    };

}
#endif