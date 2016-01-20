#ifndef __VIZIA_INPUT_H__
#define __VIZIA_INPUT_H__

void Vizia_Command(char * cmd);

bool Vizia_CommmandFilter(const char *cmd);

int Vizia_AxisFilter(int button, int value);

void Vizia_AddAxisBT(int button, int value);

char* Vizia_BTToCommand(int button);

bool Vizia_HasCounterBT(int button);

int Vizia_CounterBT(int button);

void Vizia_AddBTCommand(int button, int state);

void Vizia_InputInit();

void Vizia_InputTic();

void Vizia_InputClose();

#endif
