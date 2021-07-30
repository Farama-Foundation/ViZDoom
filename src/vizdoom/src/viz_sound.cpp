#include "sounddef.h"

#include "viz_sound.h"
#include "viz_labels.h"

EXTERN_CVAR (Int, viz_samp_freq)

int VIZ_AudioSamplesPerTic() {
    return *viz_samp_freq / TICRATE;
}

int VIZ_AudioSizePerTicBytes() {
    return SOUND_NUM_CHANNELS * sizeof(short) * VIZ_AudioSamplesPerTic();
}

int VIZ_AudioBufferSizeBytes() {
    return VIZ_AudioSizePerTicBytes() * MAX_SOUND_FRAMES_TO_STORE;
}

