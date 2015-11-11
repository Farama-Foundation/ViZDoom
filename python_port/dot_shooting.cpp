
#include <ctime>
#include <cstdlib>
#include <cstdio>

int _x;
int _y;
int _random_background;
int _max_moves;
float _living_reward;
float _miss_penalty;
float _hit_reward;
int _ammo;

int _number_of_actions;
float *_state;
int * _state_format;
int _moves_made;
int _current_ammo;
float* _current_ammo_float;
int _finished;
float _summary_reward;
int _aim_y;
int _aim_x;

int initialized = 0;

//workaround, numpy doesn't cope with not scatteres arrays : ( . . .
//separate class for contigous 2d array would be better

#define zbigniew(Y, X)  ((Y)*_x+(X))


void init(int x, int y, int random_bg,int max_moves,float living_reward,float miss_penalty, float hit_reward, int ammo)/////////////////
{
	srand(time(0));
	_finished = 1;
	_x = x;
	_y = y;
	_aim_y = _y/2;
	_aim_x = 0;
	_random_background = random_bg;


	_max_moves = max_moves;
	_living_reward = living_reward;
	_miss_penalty = miss_penalty; 
	_hit_reward = hit_reward;
	_ammo = ammo;

	if (_ammo > 0)
	{
		_state_format = new int[4];
		_state_format[0] = 3;
		_state_format[3] = 1;
	}
	else
	{
		_state_format = new int[3];
		_state_format[0] = 2;
	}
	_state_format[1] = _y;
	_state_format[2] = _x;

	_number_of_actions = 3;
	_moves_made = 0;
	_current_ammo_float = new float[1];
	_summary_reward = 0;

	if (initialized and _state)
	{

		delete[] _state;
	}
	_state = new float[_y *_x];
	for (int i =0; i<_y*_x;i++)
	{
		_state[i] = 0.0;
	}
	initialized = 1;
}
int is_finished()
{
	
	return _finished;
}
float get_summary_reward()
{
	return _summary_reward;
}
void new_episode()  
{
	_finished = 0;
	_summary_reward = 0;
	_moves_made = 0;
	
	
	if (_ammo > 0)
	{
		_current_ammo = _ammo;
		_current_ammo_float[0] = 1.0;
	}
	_state[zbigniew(_aim_y,_aim_x)] = 0.0;
	_aim_x = rand() % _x ;

	if(_random_background)
	{
		for(int i =0;i<_y*_x;i++)
		{
			//skip the "aiming line"
			if( i ==_aim_y*_x)
			{
				i += _x -1;
				continue;
			}
			_state[i]=rand()/float(RAND_MAX);
		}
	}
	
	_state[zbigniew(_aim_y,_aim_x)] = 1.0;

}
float make_action(int const* action)
{
	
	if(_finished)
	{
		//THROW something?
		return NULL;
	}

	float reward = _living_reward;
	++_moves_made; 
	if(action[0] && ! action[1])
	{
		if (_aim_x >0)
		{
			_state[zbigniew(_aim_y,_aim_x)] = 0.0;
			_aim_x -= 1;
			_state[zbigniew(_aim_y,_aim_x)] = 1.0;	
		}
	}
	else if(action[1] && ! action[0])
	{
		if (_aim_x < _x -1)
		{
			_state[zbigniew(_aim_y,_aim_x)] = 0.0;
			_aim_x += 1;
			_state[zbigniew(_aim_y,_aim_x)] = 1.0;	
		}
	}
	if (action[2])
	{
		if(_current_ammo >0)
		{
			if (_aim_x != _x/2)
			{
				reward -= _miss_penalty;
			}
			else
			{
				reward += _hit_reward;
				_finished = 1;
			}
			--_current_ammo;
		}
		_current_ammo_float[0] = _current_ammo / float(_ammo);


	}

	_summary_reward += reward;
	if (_moves_made >= _max_moves)
	{
		_finished = 1;
	}
	return reward;
}
float* get_image_state() 
{
	return _state;
}
float* get_misc_state() 
{
	return _current_ammo_float;
}
float average_best_result()
{
	float best = _hit_reward + _living_reward;
	float worst = _hit_reward + _living_reward *(_x-1)/2.0;
	float avg = (best+worst)/2.0;
	float r = (best +(_x -1)*avg)/float(_x);
	return r;
}

int get_action_format()
{
	return _number_of_actions;
}
int * get_state_format()
{
	return _state_format;
}
