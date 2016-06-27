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

#include "viz_message_queue.h"
#include "viz_input.h"
#include "viz_game.h"
#include "viz_main.h"

EXTERN_CVAR (Bool, viz_async)

bip::message_queue *vizMQController;
bip::message_queue *vizMQDoom;
char * vizMQControllerName;
char * vizMQDoomName;

void VIZ_MQInit(const char * id){

    Printf("VIZ_MQInit: Init message queues.\n");

	vizMQControllerName = new char[strlen(VIZ_MQ_NAME_CTR_BASE) + strlen(id) + 1];
	strcpy(vizMQControllerName, VIZ_MQ_NAME_CTR_BASE);
	strcat(vizMQControllerName, id);

	vizMQDoomName = new char[strlen(VIZ_MQ_NAME_DOOM_BASE) + strlen(id) +1];
	strcpy(vizMQDoomName, VIZ_MQ_NAME_DOOM_BASE);
	strcat(vizMQDoomName, id);

    try{
        vizMQController = new bip::message_queue(bip::open_only, vizMQControllerName);//, VIZ_MQ_MAX_MSG_NUM, VIZ_MQ_MAX_MSG_SIZE);
        vizMQDoom = new bip::message_queue(bip::open_only, vizMQDoomName);//, VIZ_MQ_MAX_MSG_NUM, VIZ_MQ_MAX_MSG_SIZE);
    }
    catch(...){ // bip::interprocess_exception
        Printf("VIZ_MQInit: Failed to open message queues.");
        if(vizMQController) VIZ_MQSend(VIZ_MSG_CODE_DOOM_ERROR, "Failed to open message queues.");
        exit(1);
    }
}

void VIZ_MQSend(uint8_t code){
    VIZMessageSignal msg;
    msg.code = code;
    vizMQController->send(&msg, sizeof(VIZMessageSignal), 0);
}

void VIZ_MQSend(uint8_t code, const char * command){
    VIZMessageCommand msg;
    msg.code = code;
    strncpy(msg.command, command, VIZ_MQ_MAX_CMD_LEN);
    vizMQController->send(&msg, sizeof(VIZMessageCommand), 0);
}

void VIZ_MQRecv(void *msg, size_t &size, unsigned int &priority){
    vizMQDoom->receive(msg, sizeof(VIZMessageCommand), size, priority);
}

bool VIZ_MQTryRecv(void *msg, size_t &size, unsigned int &priority){
    return vizMQDoom->try_receive(msg, sizeof(VIZMessageCommand), size, priority);
}

void VIZ_MQTic(){

    VIZMessageCommand msg;

    unsigned int priority;
    bip::message_queue::size_type recv_size;

    do {
        if(!*viz_async) VIZ_MQRecv(&msg, recv_size, priority);
        else{
            bool isMsg = VIZ_MQTryRecv(&msg, recv_size, priority);
            if(!isMsg) break;
        }
        switch(msg.code){
            case VIZ_MSG_CODE_TIC :
                vizNextTic = true;
                break;

            case VIZ_MSG_CODE_UPDATE:
                VIZ_Update();
                VIZ_GameStateTic();
                VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
                break;

            case VIZ_MSG_CODE_TIC_AND_UPDATE:
                vizUpdate = true;
                vizNextTic = true;
                break;

            case VIZ_MSG_CODE_COMMAND :
                VIZ_Command(strdup(msg.command));
                break;

            case VIZ_MSG_CODE_CLOSE :
            case VIZ_MSG_CODE_ERROR:
                exit(0);

            default : break;
        }
    }while(!vizNextTic);
}

void VIZ_MQClose(){
    //bip::message_queue::remove(vizMQControllerName);
    //bip::message_queue::remove(vizMQDoomName);
	delete[] vizMQControllerName;
	delete[] vizMQDoomName;
}