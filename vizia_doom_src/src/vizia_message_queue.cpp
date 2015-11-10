#include "vizia_message_queue.h"

#include "interprocess/ipc/message_queue.hpp"

#include "d_main.h"
#include "d_event.h"
#include "c_bind.h"
#include "c_console.h"
#include "c_dispatch.h"

#include <string.h>

bip::message_queue *viziaMQ;

void Vizia_MQInit(){
    bip::message_queue::remove(VIZIA_MQ_NAME);
    viziaMQ = new bip::message_queue(bip::open_or_create, VIZIA_MQ_NAME, VIZIA_MQ_MAX_MSG_NUM, VIZIA_MQ_MAX_MSG_SIZE);
}

void Vizia_MQSend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    viziaMQ->send(&msg, sizeof(ViziaMessageSignalStruct), 0);
    //return viziaMQ->try_send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

bool Vizia_MQTrySend(uint8_t code){
    ViziaMessageSignalStruct msg;
    msg.code = code;
    return viziaMQ->try_send(&msg, sizeof(ViziaMessageSignalStruct), 0);
}

void Vizia_MQSend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    viziaMQ->send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

bool Vizia_MQTrySend(uint8_t code, const char * command){
    ViziaMessageCommandStruct msg;
    msg.code = code;
    strncpy(msg.command, command, VIZIA_MQ_MAX_CMD_LEN);
    return viziaMQ->try_send(&msg, sizeof(ViziaMessageCommandStruct), 0);
}

void Vizia_MQRecv(void *msg, unsigned long &size, unsigned int &priority){
    viziaMQ->receive(&msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

bool Vizia_MQTryRecv(void *msg, unsigned long &size, unsigned int &priority){
    return viziaMQ->try_receive(&msg, sizeof(ViziaMessageCommandStruct), size, priority);
}

void Vizia_MQTic(){
    ViziaMessageCommandStruct msg;

    unsigned int priority;
    bip::message_queue::size_type recvd_size;

    bool nextTic = false;
    do {
        viziaMQ->receive(&msg, sizeof(ViziaMessageCommandStruct), recvd_size, priority);
        switch(msg.code){
            case VIZIA_MSG_CODE_READY :
            case VIZIA_MSG_CODE_TIC :
                nextTic = true;
                break;
            case VIZIA_MSG_CODE_COMMAND :
                AddCommandString(strdup(msg.command));
                break;
            default : break;
        }
    }while(!nextTic);
}

void Vizia_MQClose(){
    bip::message_queue::remove(VIZIA_MQ_NAME);
}