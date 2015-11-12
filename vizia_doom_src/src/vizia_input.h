#ifndef __VIZIA_INPUT_H__
#define __VIZIA_INPUT_H__

void Vizia_MouseEvent(int x, int y);

void Vizia_ButtonEvent(int button, bool state, bool oldState);

int Vizia_CounterBT(int button);

char* Vizia_BTToCommand(int button, bool state);

char* Vizia_GetCommandWithState(char* command, bool state);

void Vizia_ButtonCommand(int button, bool state, bool oldState);

void Vizia_InputInit();

void Vizia_InputTic();

void Vizia_InputClose();

#endif
