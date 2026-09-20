#include <stdio.h>
#include "fmod.h"
#include "fmod_output.h"
//VIZDOOM_CODE
#include <SDL3/SDL.h>

#define CONVERTBUFFER_SIZE 	4096	// in bytes

#define D(x)

#define FALSE 	0
#define TRUE 	1

typedef int BOOL;

//VIZDOOM_CODE
struct AudioData
{
	FMOD_OUTPUT_STATE *Output;
	SDL_AudioStream *Stream;
	int BytesPerSample;
};

//VIZDOOM_CODE
FMOD_SOUND_FORMAT Format_SDLtoFMOD(SDL_AudioFormat format)
{
	if (format == SDL_AUDIO_S8)
	{
		return FMOD_SOUND_FORMAT_PCM8;
	}
	return FMOD_SOUND_FORMAT_PCM16;
}

//VIZDOOM_CODE
SDL_AudioFormat Format_FMODtoSDL(FMOD_SOUND_FORMAT format)
{
	switch (format)
	{
	case FMOD_SOUND_FORMAT_PCM8:	return SDL_AUDIO_S8;
	case FMOD_SOUND_FORMAT_PCM16:	return SDL_AUDIO_S16;
	default: 						return SDL_AUDIO_S16;
	}
}

//VIZDOOM_CODE
static void SDLCALL AudioCallback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount)
{
	struct AudioData *data = (struct AudioData *)userdata;
	short buffer[CONVERTBUFFER_SIZE / sizeof(short)];
	while (additional_amount > 0)
	{
		int frames = (additional_amount + data->BytesPerSample - 1) / data->BytesPerSample;
		int maxframes = sizeof(buffer) / data->BytesPerSample;
		int len;
		if (frames > maxframes) frames = maxframes;
		len = frames * data->BytesPerSample;
		data->Output->readfrommixer(data->Output, buffer, frames);
		if (!SDL_PutAudioStreamData(stream, buffer, len)) break;
		additional_amount -= len;
	}
}

static FMOD_RESULT F_CALLBACK GetNumDrivers(FMOD_OUTPUT_STATE *output_state, int *numdrivers)
{
	if (numdrivers == NULL)
	{
		return FMOD_ERR_INVALID_PARAM;
	}
	*numdrivers = 1;
	return FMOD_OK;
}

static FMOD_RESULT F_CALLBACK GetDriverName(FMOD_OUTPUT_STATE *output_state, int id, char *name, int namelen)
{
	if (id != 0 || name == NULL)
	{
		return FMOD_ERR_INVALID_PARAM;
	}
	strncpy(name, "SDL default", namelen);
	return FMOD_OK;
}

static FMOD_RESULT F_CALLBACK GetDriverCaps(FMOD_OUTPUT_STATE *output_state, int id, FMOD_CAPS *caps)
{
	if (id != 0 || caps == NULL)
	{
		return FMOD_ERR_INVALID_PARAM;
	}
	*caps = FMOD_CAPS_OUTPUT_FORMAT_PCM8 | FMOD_CAPS_OUTPUT_FORMAT_PCM16 | FMOD_CAPS_OUTPUT_MULTICHANNEL;
	return FMOD_OK;
}

//VIZDOOM_CODE
static FMOD_RESULT F_CALLBACK Init(FMOD_OUTPUT_STATE *output_state, int selecteddriver,
	FMOD_INITFLAGS flags, int *outputrate, int outputchannels,
	FMOD_SOUND_FORMAT *outputformat, int dspbufferlength, int dspnumbuffers,
	void *extradriverdata)
{
	SDL_AudioSpec desired;
	struct AudioData *data;
	
	if (selecteddriver != 0 || outputrate == NULL || outputformat == NULL ||
		outputchannels <= 0 || outputchannels > CONVERTBUFFER_SIZE / 2)
	{
		D(printf("invalid param\n"));
		return FMOD_ERR_INVALID_PARAM;
	}
	if (!SDL_InitSubSystem(SDL_INIT_AUDIO))
	{
		D(printf("init subsystem failed\n"));
		return FMOD_ERR_OUTPUT_INIT;
	}
	data = malloc(sizeof(*data));
	if (data == NULL)
	{
		D(printf("nomem\n"));
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return FMOD_ERR_MEMORY;
	}
	desired.freq = *outputrate;
	desired.format = Format_FMODtoSDL(*outputformat);
	desired.channels = outputchannels;
	data->Output = output_state;
	data->Stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired, AudioCallback, data);
	if (data->Stream == NULL)
	{
		D(printf("openaudio failed\n"));
		free(data);
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return FMOD_ERR_OUTPUT_INIT;
	}
	output_state->plugindata = data;
	*outputformat = Format_SDLtoFMOD(desired.format);
	data->BytesPerSample = *outputformat == FMOD_SOUND_FORMAT_PCM16 ? 2 : 1;
	data->BytesPerSample *= desired.channels;
	D(printf("init ok\n"));
	if (!SDL_ResumeAudioStreamDevice(data->Stream))
	{
		SDL_DestroyAudioStream(data->Stream);
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		free(data);
		output_state->plugindata = NULL;
		return FMOD_ERR_OUTPUT_INIT;
	}
	return FMOD_OK;
}

//VIZDOOM_CODE
static FMOD_RESULT F_CALLBACK Close(FMOD_OUTPUT_STATE *output_state)
{
	struct AudioData *data = (struct AudioData *)output_state->plugindata;
	
	D(printf("Close\n"));
	if (data != NULL)
	{
		SDL_DestroyAudioStream(data->Stream);
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		free(data);
	}
	return FMOD_OK;
}

static FMOD_RESULT F_CALLBACK Update(FMOD_OUTPUT_STATE *update)
{
	return FMOD_OK;
}

static FMOD_RESULT F_CALLBACK GetHandle(FMOD_OUTPUT_STATE *output_state, void **handle)
{
	D(printf("Gethandle\n"));
	// SDL's audio isn't multi-instanced, so this is pretty meaningless
	if (handle == NULL)
	{
		return FMOD_ERR_INVALID_PARAM;
	}
	*handle = output_state->plugindata;
	return FMOD_OK;
}

static FMOD_OUTPUT_DESCRIPTION Desc =
{
	"SDL Output",		// name
	1,					// version
	0,					// polling
	GetNumDrivers,
	GetDriverName,
	GetDriverCaps,
	Init,
	Close,
	Update,
	GetHandle,
	NULL,				// getposition
	NULL,				// lock
	NULL				// unlock
};

F_DECLSPEC F_DLLEXPORT FMOD_OUTPUT_DESCRIPTION * F_API FMODGetOutputDescription()
{
	return &Desc;
}
