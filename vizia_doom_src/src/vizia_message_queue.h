#ifndef __VIZIA_MESSAGE_QUEUE_H__
#define __VIZIA_MESSAGE_QUEUE_H__

#include <boost/interprocess/ipc/message_queue.hpp>

namespace bip = boost::interprocess;

extern bip::message_queue *viziaMQSend;
extern bip::message_queue *viziaMQRecv;

#define VIZIA_MQ_NAME_CTR_BASE "ViziaMQCtr"
#define VIZIA_MQ_NAME_DOOM_BASE "ViziaMQDoom"
#define VIZIA_MQ_MAX_MSG_NUM 64
#define VIZIA_MQ_MAX_MSG_SIZE sizeof(ViziaMessageCommandStruct)
#define VIZIA_MQ_MAX_CMD_LEN 64

#define VIZIA_MSG_CODE_DOOM_DONE 11
#define VIZIA_MSG_CODE_DOOM_CLOSE 12
#define VIZIA_MSG_CODE_DOOM_ERROR 13

#define VIZIA_MSG_CODE_TIC 21
#define VIZIA_MSG_CODE_UPDATE 22
#define VIZIA_MSG_CODE_TIC_N_UPDATE 23
#define VIZIA_MSG_CODE_COMMAND 24
#define VIZIA_MSG_CODE_CLOSE 25
#define VIZIA_MSG_CODE_ERROR 26

struct ViziaMessageSignalStruct{
    uint8_t code;
};

struct ViziaMessageCommandStruct{
    uint8_t code;
    char command[VIZIA_MQ_MAX_CMD_LEN];
};

void Vizia_MQInit(const char * id);

void Vizia_MQSend(uint8_t code);
void Vizia_MQSend(uint8_t code, const char * command);
void Vizia_MQRecv(void *msg, unsigned long &size, unsigned int &priority);
bool Vizia_MQTryRecv(void *msg, unsigned long &size, unsigned int &priority);

void Vizia_MQTic();

void Vizia_MQClose();

#endif
