/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include "vizdoom_shared_memory.h"
#include "vizdoom_message_queue.h"
#include "vizdoom_defines.h"

#include "doomstat.h"
#include "v_video.h"

bip::shared_memory_object vizdoomSM;
size_t vizdoomSMSize;
char * vizdoomSMName;

void ViZDoom_SMInit(const char * id){

	vizdoomSMName = new char[strlen(VIZDOOM_SM_NAME_BASE) + strlen(id)];
	strcpy(vizdoomSMName, VIZDOOM_SM_NAME_BASE);
	strcat(vizdoomSMName, id);

    try {
        bip::shared_memory_object::remove(vizdoomSMName);
        vizdoomSM = bip::shared_memory_object(bip::open_or_create, vizdoomSMName, bip::read_write);

        vizdoomSMSize = sizeof(ViZDoomInputStruct) + sizeof(ViZDoomGameVarsStruct) +
                      (sizeof(BYTE) * screen->GetWidth() * screen->GetHeight() * 4);
        vizdoomSM.truncate(vizdoomSMSize);
    }
    catch(bip::interprocess_exception &ex){
        printf("ViZDoom_SMInit: Error creating shared memory");
        ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_ERROR);
        exit(1);
    }
}

void ViZDoom_SMClose(){
    bip::shared_memory_object::remove(vizdoomSMName);
	delete[] vizdoomSMName;
}