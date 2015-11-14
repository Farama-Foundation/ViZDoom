#ifndef __VIZIA_MESSAGE_QUEUE_H__
#define __VIZIA_MESSAGE_QUEUE_H__

#include <boost/interprocess/ipc/message_queue.hpp>

namespace bip = boost::interprocess;

extern bip::message_queue *viziaMQSend;
extern bip::message_queue *viziaMQRecv;

#define VIZIA_MQ_NAME_CTR "ViziaMQCtr"
#define VIZIA_MQ_NAME_DOOM "ViziaMQDoom"
#define VIZIA_MQ_MAX_MSG_NUM 32
#define VIZIA_MQ_MAX_MSG_SIZE sizeof(ViziaMessageCommandStruct)
#define VIZIA_MQ_MAX_CMD_LEN 32

#define VIZIA_MSG_CODE_DOOM_READY 10
#define VIZIA_MSG_CODE_DOOM_TIC 11
#define VIZIA_MSG_CODE_DOOM_CLOSE 12

#define VIZIA_MSG_CODE_READY 0
#define VIZIA_MSG_CODE_TIC 1
#define VIZIA_MSG_CODE_CLOSE 2
#define VIZIA_MSG_CODE_COMMAND 3

struct ViziaMessageSignalStruct{
    uint8_t code;
};

struct ViziaMessageCommandStruct{
    uint8_t code;
    char command[VIZIA_MQ_MAX_CMD_LEN];
};

void Vizia_Command(char * command);

void Vizia_MQInit();

void Vizia_MQSend(uint8_t code);

bool Vizia_MQTrySend(uint8_t code);

void Vizia_MQSend(uint8_t code, const char * command);

bool Vizia_MQTrySend(uint8_t code, const char * command);

void Vizia_MQRecv(void *msg, unsigned long &size, unsigned int &priority);

bool Vizia_MQTryRecv(void *msg, unsigned long &size, unsigned int &priority);

void Vizia_MQTic();

void Vizia_MQClose();

#endif
