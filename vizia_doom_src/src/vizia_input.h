#ifndef __VIZIA_INPUT_H__
#define __VIZIA_INPUT_H__

void Vizia_Command(char * command);

bool Vizia_CommmandFilter(const char *cmd);

char* Vizia_BTToCommand(int button);

bool Vizia_HasCounterBT(int button);

int Vizia_CounterBT(int button);

void Vizia_AddBTCommand(char* command, bool state);

void Vizia_Mouse(int x, int y);

void Vizia_InputInit();

void Vizia_InputTic();

void Vizia_InputClose();

#endif
