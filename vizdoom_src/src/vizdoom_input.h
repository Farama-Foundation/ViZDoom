#ifndef __VIZDOOM_INPUT_H__
#define __VIZDOOM_INPUT_H__

void ViZDoom_Command(char * cmd);

bool ViZDoom_CommmandFilter(const char *cmd);

int ViZDoom_AxisFilter(int button, int value);

void ViZDoom_AddAxisBT(int button, int value);

char* ViZDoom_BTToCommand(int button);

void ViZDoom_ResetDiscontinuousBT();

void ViZDoom_AddBTCommand(int button, int state);

void ViZDoom_InputInit();

void ViZDoom_InputTic();

void ViZDoom_InputClose();

#endif
