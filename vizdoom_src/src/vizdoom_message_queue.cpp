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

#include "vizdoom_message_queue.h"
#include "vizdoom_input.h"
#include "vizdoom_game.h"
#include "vizdoom_main.h"
#include "vizdoom_defines.h"

EXTERN_CVAR (Bool, vizdoom_async)

bip::message_queue *vizdoomMQController;
bip::message_queue *vizdoomMQDoom;
char * vizdoomMQControllerName;
char * vizdoomMQDoomName;

void ViZDoom_MQInit(const char * id){

	vizdoomMQControllerName = new char[strlen(VIZDOOM_MQ_NAME_CTR_BASE) + strlen(id)];
	strcpy(vizdoomMQControllerName, VIZDOOM_MQ_NAME_CTR_BASE);
	strcat(vizdoomMQControllerName, id);

	vizdoomMQDoomName = new char[strlen(VIZDOOM_MQ_NAME_DOOM_BASE) + strlen(id)];
	strcpy(vizdoomMQDoomName, VIZDOOM_MQ_NAME_DOOM_BASE);
	strcat(vizdoomMQDoomName, id);

    try{
        vizdoomMQController = new bip::message_queue(bip::open_only, vizdoomMQControllerName);//, VIZDOOM_MQ_MAX_MSG_NUM, VIZDOOM_MQ_MAX_MSG_SIZE);
        vizdoomMQDoom = new bip::message_queue(bip::open_only, vizdoomMQDoomName);//, VIZDOOM_MQ_MAX_MSG_NUM, VIZDOOM_MQ_MAX_MSG_SIZE);
    }
    catch(bip::interprocess_exception &ex){
        Printf("ViZDoom_MQInit: Error creating message queues");
        ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_ERROR);
        exit(1);
    }
}

void ViZDoom_MQSend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    vizdoomMQController->send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

void ViZDoom_MQSend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZDOOM_MQ_MAX_CMD_LEN);
    vizdoomMQController->send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

void ViZDoom_MQRecv(void *msg, size_t &size, unsigned int &priority){
    vizdoomMQDoom->receive(msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

bool ViZDoom_MQTryRecv(void *msg, size_t &size, unsigned int &priority){
    return vizdoomMQDoom->try_receive(msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

void ViZDoom_MQTic(){

    ViziaMessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recv_size;

    do {
        if(!*vizdoom_async) ViZDoom_MQRecv(&msg, recv_size, priority);
        else{
            bool isMsg = ViZDoom_MQTryRecv(&msg, recv_size, priority);
            if(!isMsg) break;
        }
        switch(msg.code){
            case VIZDOOM_MSG_CODE_TIC :
                vizdoomNextTic = true;
                break;

            case VIZDOOM_MSG_CODE_UPDATE:
                ViZDoom_Update();
                ViZDoom_GameVarsTic();
                ViZDoom_MQSend(VIZDOOM_MSG_CODE_DOOM_DONE);
                break;

            case VIZDOOM_MSG_CODE_TIC_N_UPDATE:
                vizdoomUpdate = true;
                vizdoomNextTic = true;
                break;

            case VIZDOOM_MSG_CODE_COMMAND :
                ViZDoom_Command(strdup(msg.command));
                break;

            case VIZDOOM_MSG_CODE_CLOSE :
            case VIZDOOM_MSG_CODE_ERROR:
                exit(0);

            default : break;
        }
    }while(!vizdoomNextTic);
}

void ViZDoom_MQClose(){
    //bip::message_queue::remove(vizdoomMQControllerName);
    //bip::message_queue::remove(vizdoomMQDoomName);
	delete[] vizdoomMQControllerName;
	delete[] vizdoomMQDoomName;
}