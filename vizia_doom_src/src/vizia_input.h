#ifndef __VIZIA_INPUT_H__
#define __VIZIA_INPUT_H__

bool Vizia_HasCounterBT(int button);

int Vizia_CounterBT(int button);

char* Vizia_BTToCommand(int button, bool state);

char* Vizia_GetCommandWithState(char* command, bool state);

void Vizia_Mouse(int x, int y);

void Vizia_InputInit();

void Vizia_InputTic();

void Vizia_InputClose();

#endif
