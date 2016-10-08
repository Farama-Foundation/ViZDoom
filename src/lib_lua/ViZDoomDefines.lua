local Mode = {}

Mode.PLAYER             = 0 -- synchronous player mode
Mode.SPECTATOR          = 1 -- synchronous spectator mode
Mode.ASYNC_PLAYER       = 2 -- asynchronous player mode
Mode.ASYNC_SPECTATOR    = 3 -- asynchronous spectator mode

vizdoom.Mode = Mode

local ScreenResolution = {}

ScreenResolution.RES_160X120 = 0    -- 4:3

ScreenResolution.RES_200X125 = 1    -- 16:10
ScreenResolution.RES_200X150 = 2    -- 4:3

ScreenResolution.RES_256X144 = 3    -- 16:9
ScreenResolution.RES_256X160 = 4    -- 16:10
ScreenResolution.RES_256X192 = 5    -- 4:3

ScreenResolution.RES_320X180 = 6    -- 16:9
ScreenResolution.RES_320X200 = 7    -- 16:10
ScreenResolution.RES_320X240 = 8    -- 4:3
ScreenResolution.RES_320X256 = 9    -- 5:4

ScreenResolution.RES_400X225 = 10    -- 16:9
ScreenResolution.RES_400X250 = 11    -- 16:10
ScreenResolution.RES_400X300 = 12    -- 4:3

ScreenResolution.RES_512X288 = 13    -- 16:9
ScreenResolution.RES_512X320 = 14    -- 16:10
ScreenResolution.RES_512X384 = 15    -- 4:3

ScreenResolution.RES_640X360 = 16    -- 16:9
ScreenResolution.RES_640X400 = 17    -- 16:10
ScreenResolution.RES_640X480 = 18    -- 4:3

ScreenResolution.RES_800X450 = 19    -- 16:9
ScreenResolution.RES_800X500 = 20    -- 16:10
ScreenResolution.RES_800X600 = 21    -- 4:3

ScreenResolution.RES_1024X576 = 22   -- 16:9
ScreenResolution.RES_1024X640 = 23   -- 16:10
ScreenResolution.RES_1024X768 = 24   -- 4:3

ScreenResolution.RES_1280X720 = 25   -- 16:9
ScreenResolution.RES_1280X800 = 26   -- 16:10
ScreenResolution.RES_1280X960 = 27   -- 4:3
ScreenResolution.RES_1280X1024 = 28  -- 5:4

ScreenResolution.RES_1400X787 = 29   -- 16:9
ScreenResolution.RES_1400X875 = 30   -- 16:10
ScreenResolution.RES_1400X1050 = 31  -- 4:3

ScreenResolution.RES_1600X900 = 32   -- 16:9
ScreenResolution.RES_1600X1000 = 33  -- 16:10
ScreenResolution.RES_1600X1200 = 34  -- 4:3

ScreenResolution.RES_1920X1080 = 35  -- 16:9

vizdoom.ScreenResolution = ScreenResolution


local ScreenFormat = {}

ScreenFormat.CRCGCB              = 0
ScreenFormat.CRCGCBDB            = 1
ScreenFormat.RGB24               = 2
ScreenFormat.RGBA32              = 3
ScreenFormat.ARGB32              = 4
ScreenFormat.CBCGCR              = 5
ScreenFormat.CBCGCRDB            = 6
ScreenFormat.BGR24               = 7
ScreenFormat.BGRA32              = 8
ScreenFormat.ABGR32              = 9
ScreenFormat.GRAY8               = 10
ScreenFormat.DEPTH_BUFFER8       = 11
ScreenFormat.DOOM_256_COLORS8    = 12

vizdoom.ScreenFormat = ScreenFormat


local GameVariable = {}

GameVariable.KILLCOUNT            = 0
GameVariable.ITEMCOUNT            = 1
GameVariable.SECRETCOUNT          = 2
GameVariable.FRAGCOUNT            = 3
GameVariable.DEATHCOUNT           = 4
GameVariable.HEALTH               = 5
GameVariable.ARMOR                = 6
GameVariable.DEAD                 = 7
GameVariable.ON_GROUND            = 8
GameVariable.ATTACK_READY         = 9
GameVariable.ALTATTACK_READY      = 10
GameVariable.SELECTED_WEAPON      = 11
GameVariable.SELECTED_WEAPON_AMMO = 12
GameVariable.AMMO0                = 13
GameVariable.AMMO1                = 14
GameVariable.AMMO2                = 15
GameVariable.AMMO3                = 16
GameVariable.AMMO4                = 17
GameVariable.AMMO5                = 18
GameVariable.AMMO6                = 19
GameVariable.AMMO7                = 20
GameVariable.AMMO8                = 21
GameVariable.AMMO9                = 22
GameVariable.WEAPON0              = 23
GameVariable.WEAPON1              = 24
GameVariable.WEAPON2              = 25
GameVariable.WEAPON3              = 26
GameVariable.WEAPON4              = 27
GameVariable.WEAPON5              = 28
GameVariable.WEAPON6              = 29
GameVariable.WEAPON7              = 30
GameVariable.WEAPON8              = 31
GameVariable.WEAPON9              = 32
GameVariable.USER1                = 33
GameVariable.USER2                = 34
GameVariable.USER3                = 35
GameVariable.USER4                = 36
GameVariable.USER5                = 37
GameVariable.USER6                = 38
GameVariable.USER7                = 39
GameVariable.USER8                = 40
GameVariable.USER9                = 41
GameVariable.USER10               = 42
GameVariable.USER11               = 43
GameVariable.USER12               = 44
GameVariable.USER13               = 45
GameVariable.USER14               = 46
GameVariable.USER15               = 47
GameVariable.USER16               = 48
GameVariable.USER17               = 49
GameVariable.USER18               = 50
GameVariable.USER19               = 51
GameVariable.USER20               = 52
GameVariable.USER21               = 53
GameVariable.USER22               = 54
GameVariable.USER23               = 55
GameVariable.USER24               = 56
GameVariable.USER25               = 57
GameVariable.USER26               = 58
GameVariable.USER27               = 59
GameVariable.USER28               = 60
GameVariable.USER29               = 61
GameVariable.USER30               = 62
GameVariable.PLAYER_NUMBER        = 63
GameVariable.PLAYER_COUNT         = 64
GameVariable.PLAYER1_FRAGCOUNT    = 65
GameVariable.PLAYER2_FRAGCOUNT    = 66
GameVariable.PLAYER3_FRAGCOUNT    = 67
GameVariable.PLAYER4_FRAGCOUNT    = 68
GameVariable.PLAYER5_FRAGCOUNT    = 69
GameVariable.PLAYER6_FRAGCOUNT    = 70
GameVariable.PLAYER7_FRAGCOUNT    = 71
GameVariable.PLAYER8_FRAGCOUNT    = 72

vizdoom.GameVariable = GameVariable

local Button = {}

Button.ATTACK          = 0
Button.USE             = 1
Button.JUMP            = 2
Button.CROUCH          = 3
Button.TURN180         = 4
Button.ALTATTACK       = 5
Button.RELOAD          = 6
Button.ZOOM            = 7

Button.SPEED           = 8
Button.STRAFE          = 9

Button.MOVE_RIGHT      = 10
Button.MOVE_LEFT       = 11
Button.MOVE_BACKWARD   = 12
Button.MOVE_FORWARD    = 13
Button.TURN_RIGHT      = 14
Button.TURN_LEFT       = 15
Button.LOOK_UP         = 16
Button.LOOK_DOWN       = 17
Button.MOVE_UP         = 18
Button.MOVE_DOWN       = 19
Button.LAND            = 20
Button.SELECT_WEAPON1  = 21
Button.SELECT_WEAPON2  = 22
Button.SELECT_WEAPON3  = 23
Button.SELECT_WEAPON4  = 24
Button.SELECT_WEAPON5  = 25
Button.SELECT_WEAPON6  = 26
Button.SELECT_WEAPON7  = 27
Button.SELECT_WEAPON8  = 28
Button.SELECT_WEAPON9  = 29
Button.SELECT_WEAPON0  = 30
Button.SELECT_NEXT_WEAPON          = 31
Button.SELECT_PREV_WEAPON          = 32
Button.DROP_SELECTED_WEAPON        = 33
Button.ACTIVATE_SELECTED_ITEM      = 34
Button.SELECT_NEXT_ITEM            = 35
Button.SELECT_PREV_ITEM            = 36
Button.DROP_SELECTED_ITEM          = 37
Button.LOOK_UP_DOWN_DELTA          = 38
Button.TURN_LEFT_RIGHT_DELTA       = 39
Button.MOVE_FORWARD_BACKWARD_DELTA = 40
Button.MOVE_LEFT_RIGHT_DELTA       = 41
Button.MOVE_UP_DOWN_DELTA          = 42

vizdoom.Button = Button
