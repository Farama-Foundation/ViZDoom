struct VIZIA_State
{
	float* image;
	float* misc;
};
struct VIZIA_StateFormat
{
	// number of shapes
	int image_shape_len;
	// array of length image_shape_len
	int* image_shape;
	// miscallanous vector length
	int misc_len;
};
struct VIZIA_ActionFormat
{
	// number of supported keys
	int keys_num;
	// wheter mouse delta is supported or not
	bool mouse_delta;
};



void init(int x, int y, int random_bg,int max_moves,double living_reward,double miss_penalty, double hit_reward, int ammo);

VIZIA_StateFormat* get_state_format();
int get_action_format();

void new_episode();
double make_action( int const* action);
int is_finished();
double get_summary_reward();
VIZIA_State* get_state();


double average_best_result();

