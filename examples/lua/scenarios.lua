game = DoomGame()

game:load_config("../../examples/config/basic.cfg")
--game:load_config("../../examples/config/deadly_corridor.cfg")
--game:load_config("../../examples/config/deathmatch.cfg")
--game:load_config("../../examples/config/defend_the_center.cfg")
--game:load_config("../../examples/config/defend_the_line.cfg")
--game:load_config("../../examples/config/health_gathering.cfg")
--game:load_config("../../examples/config/my_way_home.cfg")
--game:load_config("../../examples/config/predict_position.cfg")
--game:load_config("../../examples/config/take_cover.cfg")

game:set_screen_resolution(DoomGame.RES_640X480)
game:init()


actions_num = game:get_available_buttons_size()
actions = {}
for i = 1, actions_num do
    actions[i] = {}
    for j = 1, actions_num do
        actions[i][j] = 0
        if i == j then actions[i][j] = 1 end
    end
end

print(actions)

episodes = 10
sleep_time = 28

for i = 1, episodes do

    print("Episode #" .. i)

    game:new_episode()

    while not game:is_episode_finished() do

        s = game:get_state()
        r = game:make_action(actions[math.random(1,actions_num)])

        print("State # " .. s.number)
        print("Reward: " .. r)
        print("=====================")

        if sleep_time > 0 then
            sleep(sleep_time)
        end
    end

    print("Episode finished.")
    print("total reward: " .. game:get_total_reward())
    print("************************")

end

game:close()

