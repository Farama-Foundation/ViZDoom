
// Moved from sdl/i_system.cpp

#include <string.h>

//VIZDOOM_CODE
#include <SDL3/SDL.h>

#include "bitmap.h"
#include "v_palette.h"
#include "textures.h"

//VIZDOOM_CODE
bool I_SetCursor(FTexture *cursorpic)
{
	static SDL_Cursor *cursor;
	static SDL_Surface *cursorSurface;

	if (cursorpic != NULL && cursorpic->UseType != FTexture::TEX_Null)
	{
		// Must be no larger than 32x32.
		if (cursorpic->GetWidth() > 32 || cursorpic->GetHeight() > 32)
		{
			return false;
		}

		if (cursorSurface == NULL)
			cursorSurface = SDL_CreateSurface (32, 32, SDL_PIXELFORMAT_ARGB8888);

		SDL_LockSurface(cursorSurface);
		BYTE buffer[32*32*4];
		memset(buffer, 0, 32*32*4);
		FBitmap bmp(buffer, 32*4, 32, 32);
		cursorpic->CopyTrueColorPixels(&bmp, 0, 0);
		memcpy(cursorSurface->pixels, bmp.GetPixels(), 32*32*4);
		SDL_UnlockSurface(cursorSurface);

		if (cursor)
			SDL_DestroyCursor (cursor);
		cursor = SDL_CreateColorCursor (cursorSurface, 0, 0);
		SDL_SetCursor (cursor);
	}
	else
	{
		if (cursor)
		{
			SDL_SetCursor (SDL_GetDefaultCursor());
			SDL_DestroyCursor (cursor);
			cursor = NULL;
		}
		if (cursorSurface != NULL)
		{
			SDL_DestroySurface(cursorSurface);
			cursorSurface = NULL;
		}
	}
	return true;
}
