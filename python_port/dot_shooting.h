void init(int x, int y, int random_bg,int max_moves,float living_reward,float miss_penalty, float hit_reward, int ammo);
int * get_state_format();
int get_action_format();

void new_episode();
float make_action( int const* action);
int is_finished();
float get_summary_reward();
float * get_image_state();
float * get_misc_state();

float average_best_result();
