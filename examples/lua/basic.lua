game = DoomGame()

game:set_vizdoom_path("../../bin/vizdoom")
game:set_doom_game_path("../../scenarios/freedoom2.wad")
--game.set_doom_game_path("../../scenarios/doom2.wad")
game:set_doom_scenario_path("../../scenarios/basic.wad")
--game:set_doom_scenario_path("../../scenarios/deadly_corridor.wad")
game:set_doom_map("map01")
game:set_screen_resolution(DoomGame.RES_640X480)

game:set_render_hud(false)
game:set_render_crosshair(false)
game:set_render_weapon(true)
game:set_render_decals(false)
game:set_render_particles(false)


game:add_available_button(DoomGame.MOVE_LEFT)
game:add_available_button(DoomGame.MOVE_RIGHT)
game:add_available_button(DoomGame.ATTACK)

game:add_available_game_variable(DoomGame.AMMO2)


game:set_episode_timeout(100)
game:set_episode_start_time(10)
game:set_window_visible(true)
game:set_sound_enabled(true)
game:set_living_reward(-1)
game:set_mode(DoomGame.PLAYER)

game:init()


actions = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

episodes = 10

sleep_time = 28

for i = 1, episodes do

    print("Episode #" .. i)

    game:new_episode()

    while not game:is_episode_finished() do

        s = game:get_state()
        r = game:make_action(actions[math.random(1,3)])

        screen = s.screen_buffer
        depth = s.depth_buffer
        labels = s.labels_buffer
        map = s.map_buffer

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
