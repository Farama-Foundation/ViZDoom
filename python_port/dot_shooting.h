void init(int x, int y, int random_bg,int max_moves,double living_reward,double miss_penalty, double hit_reward, int ammo);
int is_finished();
double get_summary_reward();
void new_episode();
double make_action(int* action);
int* get_state_format();
float** get_image_state();
float* get_misc_state();
float average_best_result();