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
//not used yet because we ignore the mouse delta
struct VIZIA_ActionFormat
{
	// number of supported keys
	int keys_num;
	// wheter mouse delta is supported or not
	bool mouse_delta;
};



VIZIA_StateFormat* get_state_format();
VIZIA_State* get_state();

int get_action_format();
void new_episode();
double make_action( int const* action );
int is_finished();
void init(int x, int y, int maxtime);
void close();

//dot shoting specific
double get_summary_reward();