#include "ViziaDoomConfig.h"

namespace Vizia {
    Button getButtonId(std::string name) {
        if (name.compare("ATTACK") == 0) return ATTACK;
        else if (name.compare("USE") == 0) return USE;
        else if (name.compare("JUMP") == 0) return JUMP;
        else if (name.compare("CROUCH") == 0) return CROUCH;
        else if (name.compare("TURN180") == 0) return TURN180;
        else if (name.compare("ALTATTACK") == 0) return ALTATTACK;
        else if (name.compare("RELOAD") == 0) return RELOAD;
        else if (name.compare("ZOOM") == 0) return ZOOM;
        else if (name.compare("SPEED") == 0) return SPEED;
        else if (name.compare("STRAFE") == 0) return STRAFE;
        else if (name.compare("MOVERIGHT") == 0) return MOVE_RIGHT;
        else if (name.compare("MOVELEFT") == 0) return MOVE_LEFT;
        else if (name.compare("BACK") == 0) return MOVE_BACK;
        else if (name.compare("FORWARD") == 0) return MOVE_FORWARD;
        else if (name.compare("RIGHT") == 0) return TURN_RIGHT;
        else if (name.compare("LEFT") == 0) return TURN_LEFT;
        else if (name.compare("LOOKUP") == 0) return LOOK_UP;
        else if (name.compare("LOOKDOWN") == 0) return LOOK_DOWN;
        else if (name.compare("MOVEUP") == 0) return MOVE_UP;
        else if (name.compare("MOVEDOWN") == 0) return MOVE_DOWN;
        else if (name.compare("WEAPON1") == 0) return SELECT_WEAPON1;
        else if (name.compare("WEAPON2") == 0) return SELECT_WEAPON2;
        else if (name.compare("WEAPON3") == 0) return SELECT_WEAPON3;
        else if (name.compare("WEAPON4") == 0) return SELECT_WEAPON4;
        else if (name.compare("WEAPON5") == 0) return SELECT_WEAPON5;
        else if (name.compare("WEAPON6") == 0) return SELECT_WEAPON6;
        else if (name.compare("WEAPON7") == 0) return SELECT_WEAPON7;
        else if (name.compare("WEAPONNEXT") == 0) return SELECT_NEXT_WEAPON;
        else if (name.compare("WEAPONPREV") == 0) return SELECT_PREV_WEAPON;
        else return UNDEFINED_BUTTON;
    };

    GameVariable getGameVarId(std::string name) {
        if (name.compare("KILLCOUNT") == 0) return KILLCOUNT;
        else if (name.compare("ITEMCOUNT") == 0) return ITEMCOUNT;
        else if (name.compare("SECRETCOUNT") == 0) return SECRETCOUNT;
        else if (name.compare("HEALTH") == 0) return HEALTH;
        else if (name.compare("ARMOR") == 0) return ARMOR;
        else if (name.compare("SELECTED_WEAPON") == 0) return SELECTED_WEAPON;
        else if (name.compare("SELECTED_WEAPON_AMMO") == 0) return SELECTED_WEAPON_AMMO;
        else if (name.compare("AMMO1") == 0) return AMMO1;
        else if (name.compare("AMMO_CLIP") == 0) return AMMO1;
        else if (name.compare("AMMO2") == 0) return AMMO2;
        else if (name.compare("AMMO_SHELL") == 0) return AMMO2;
        else if (name.compare("AMMO3") == 0) return AMMO3;
        else if (name.compare("AMMO_ROCKET") == 0) return AMMO3;
        else if (name.compare("AMMO4") == 0) return AMMO4;
        else if (name.compare("AMMO_CELL") == 0) return AMMO4;
        else if (name.compare("WEAPON1") == 0) return WEAPON1;
        else if (name.compare("WEAPON_FIST") == 0) return WEAPON1;
        else if (name.compare("WEAPON_CHAINSAW") == 0) return WEAPON1;
        else if (name.compare("WEAPON2") == 0) return WEAPON2;
        else if (name.compare("WEAPON_PISTOL") == 0) return WEAPON2;
        else if (name.compare("WEAPON3") == 0) return WEAPON3;
        else if (name.compare("WEAPON_SHOTGUN") == 0) return WEAPON3;
        else if (name.compare("WEAPON_SSG") == 0) return WEAPON3;
        else if (name.compare("WEAPON_SUPER_SHOTGUN") == 0) return WEAPON3;
        else if (name.compare("WEAPON4") == 0) return WEAPON4;
        else if (name.compare("WEAPON_CHAINGUN") == 0) return WEAPON4;
        else if (name.compare("WEAPON5") == 0) return WEAPON5;
        else if (name.compare("WEAPON_ROCKET_LUNCHER") == 0) return WEAPON5;
        else if (name.compare("WEAPON6") == 0) return WEAPON6;
        else if (name.compare("WEAPON_PLASMA_GUN") == 0) return WEAPON6;
        else if (name.compare("WEAPON7") == 0) return WEAPON7;
        else if (name.compare("WEAPON_BFG") == 0) return WEAPON7;
        else if (name.compare("KEY1") == 0) return KEY1;
        else if (name.compare("KEY_BLUE") == 0) return KEY1;
        else if (name.compare("KEY2") == 0) return KEY2;
        else if (name.compare("KEY_RED") == 0) return KEY2;
        else if (name.compare("KEY3") == 0) return KEY3;
        else if (name.compare("KEY_YELLOW") == 0) return KEY3;
        else return UNDEFINED_VAR;
    }
}