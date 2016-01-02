#ifndef __VIZIA_DEFINES_H__
#define __VIZIA_DEFINES_H__

#include <exception>
#include <string>

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
        AMMO1,
        AMMO2,
        AMMO3,
        AMMO4,
        AMMO5,
        AMMO6,
        AMMO7,
        AMMO8,
        AMMO9,
        AMMO0,
        WEAPON1,
        WEAPON2,
        WEAPON3,
        WEAPON4,
        WEAPON5,
        WEAPON6,
        WEAPON7,
        WEAPON8,
        WEAPON9,
        WEAPON0,
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
    };

    static const int UserVarsNumber = 30;
    static const int SlotsNumber = 10;

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
        SELECT_WEAPON8 = 28,
        SELECT_WEAPON9 = 29,
        SELECT_WEAPON0 = 30,

        SELECT_NEXT_WEAPON = 31,
        SELECT_PREV_WEAPON = 32,
        DROP_SELECTED_WEAPON = 33,

        ACTIVATE_SELECTED_ITEM = 34,
        SELECT_NEXT_ITEM = 35,
        SELECT_PREV_ITEM = 36,
        DROP_SELECTED_ITEM = 37,

        VIEW_PITCH = 38,
        VIEW_ANGLE = 39,
    };

    static const int ButtonsNumber = 40;
    
#define VK_2_DK(vk, dk) static const std::string KEY_ ## vk = dk;
    
    VK_2_DK(TAB, "tab")
    VK_2_DK(CAPS_LOCK, "capslock")
    VK_2_DK(SHIFT, "shift")
    VK_2_DK(CTRL, "ctrl")
    VK_2_DK(ALT, "alt")
    VK_2_DK(SPACE, "space")
    VK_2_DK(ENTER, "enter")
    VK_2_DK(RIGHT_ARROW, "rightarrow")
    VK_2_DK(LEFT_ARROW, "leftarrow")
    VK_2_DK(UP_ARROW, "uparrow")
    VK_2_DK(DOWN_ARROW, "downarrow")
    VK_2_DK(INSERT, "ins")
    VK_2_DK(END, "end")
    VK_2_DK(HOME, "home")
    VK_2_DK(PAGE_UP, "pgup")
    VK_2_DK(PAGE_DOWN, "pgdn")
    VK_2_DK(F1, "f1")
    VK_2_DK(F2, "f2")
    VK_2_DK(F3, "f3")
    VK_2_DK(F4, "f4")
    VK_2_DK(F5, "f5")
    VK_2_DK(F6, "f6")
    VK_2_DK(F7, "f7")
    VK_2_DK(F8, "f8")
    VK_2_DK(F9, "f9")
    VK_2_DK(F10, "f10")
    VK_2_DK(F11, "f11")
    VK_2_DK(F12, "f12")
    
    VK_2_DK(Q, "q")
    VK_2_DK(W, "w")
    VK_2_DK(E, "e")
    VK_2_DK(R, "r")
    VK_2_DK(T, "t")
    VK_2_DK(Y, "y")
    VK_2_DK(U, "u")
    VK_2_DK(I, "i")
    VK_2_DK(O, "o")
    VK_2_DK(P, "p")
    VK_2_DK(A, "a")
    VK_2_DK(S, "s")
    VK_2_DK(D, "d")
    VK_2_DK(F, "f")
    VK_2_DK(G, "g")
    VK_2_DK(H, "h")
    VK_2_DK(J, "j")
    VK_2_DK(K, "k")
    VK_2_DK(L, "l")
    VK_2_DK(Z, "z")
    VK_2_DK(X, "x")
    VK_2_DK(C, "c")
    VK_2_DK(V, "v")
    VK_2_DK(B, "b")
    VK_2_DK(N, "n")
    VK_2_DK(M, "m")
    
    VK_2_DK(1, "1")
    VK_2_DK(2, "2")
    VK_2_DK(3, "3")
    VK_2_DK(4, "4")
    VK_2_DK(5, "5")
    VK_2_DK(6, "6")
    VK_2_DK(7, "7")
    VK_2_DK(8, "8")
    VK_2_DK(9, "9")
    VK_2_DK(0, "0")
    VK_2_DK(MINUS, "-")
    VK_2_DK(EQUALS, "Equals")
    
    VK_2_DK(COMMA, ",")
    VK_2_DK(PERIOD, ".")
    VK_2_DK(LEFT_BRACKET, "LeftBracket")
    VK_2_DK(RIGHT_BRACKET, "RightBracket")
    VK_2_DK(SLASH, "/")
    VK_2_DK(BACKSLASH, "\\")
    VK_2_DK(SEMICOLON, ";")
    VK_2_DK(QUOTE, "'")

    VK_2_DK(MOUSE_1, "mouse1")
    VK_2_DK(MOUSE_2, "mouse2")
    VK_2_DK(MOUSE_3, "mouse3")
    VK_2_DK(MOUSE_4, "mouse4")

    static const int DOOM_WEAPON_FIST = 0;
    static const int DOOM_WEAPON_CHAINSAW = 0;
    static const int DOOM_WEAPON_PISTOL = 1;
    static const int DOOM_WEAPON_SHOTGUN = 3;
    static const int DOOM_WEAPON_SUPER_SHOTGUN = 3;
    static const int DOOM_WEAPON_CHAINGUN = 4;
    static const int DOOM_WEAPON_ROCKET_LUNCHER = 5;
    static const int DOOM_WEAPON_PLASMA_GUN = 6;
    static const int DOOM_WEAPON_BFG = 7;

}
#endif