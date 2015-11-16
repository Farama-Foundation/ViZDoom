void init(int x, int y, int random_bg,int max_moves,double living_reward,double miss_penalty, double hit_reward, int ammo);

int * get_state_format();
int get_action_format();

void new_episode();
double make_action( int const* action);
int is_finished();
double get_summary_reward();
float * get_image_state();
float * get_misc_state();

double average_best_result();
