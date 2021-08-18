//
// Similar to doomdef.h, but for sound-related stuff
//

#ifndef VIZDOOM_SOUNDDEF_H
#define VIZDOOM_SOUNDDEF_H

#include "doomdef.h"

#define DEFAULT_SOUND_FREQ          44100
#define DEFAULT_SAMPLES_TIC         (DEFAULT_SOUND_FREQ / TICRATE)

#define SOUND_NUM_CHANNELS          2  // it is stereo by default

#endif //VIZDOOM_SOUNDDEF_H