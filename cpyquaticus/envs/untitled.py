import ctypes
import numpy as np

def load_cpyquaticus():
    try:
        cpyquaticus = ctypes.CDLL("./cpyquaticus.so")
    except OSError as e:
        raise RuntimeError(f"Failed to load shared library: {e}")
        


    #Start reward.h
    class team_rewards(ctypes.Structure):
        _fields_ = [
            ("team_grab", ctypes.c_int),
            ("team_capture", ctypes.c_int),
            ("opp_grab", ctypes.c_int),
            ("opp_capture", ctypes.c_int),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_entity.restype = team_rewards
    class agent_rewards(ctypes.Structure):
        _fields_ = [
            ("tag", ctypes.c_int),
            ("grab", ctypes.c_int),
            ("capture", ctypes.c_int),
            ("got_tagged", ctypes.c_int),
            ("oob", ctypes.c_int),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_entity.restype = agent_rewards

    cpyquaticus.create_team_reward.argtypes = ()
    cpyquaticus.create_team_reward.argtypes = ctypes.POINTER(team_rewards)

    cpyquaticus.create_agent_reward.argtypes = ()
    cpyquaticus.create_agent_reward.argtypes = ctypes.POINTER(agent_rewards)

    cpyquaticus.free_team_reward.argtypes = (ctypes.POINTER(team_rewards))
    cpyquaticus.free_team_reward.argtypes = None

    cpyquaticus.free_agent_reward.argtypes = (ctypes.POINTER(agent_rewards))
    cpyquaticus.free_agent_reward.argtypes = None

    #End rewards.h

    #Start heron.h
    class PID(ctypes.Structure):
        _fields_ = [
            ("dt", ctypes.c_float),
            ("kp", ctypes.c_float),
            ("ki", ctypes.c_float),
            ("kd", ctypes.c_float),
            ("integral_max", ctypes.c_float),
            ("prev_error", ctypes.c_float),
            ("integral", ctypes.c_float),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_float, 
                                            ctypes.c_float, 
                                            ctypes.c_float, 
                                            ctypes.c_float, 
                                            ctypes.c_float,
                                            ctypes.c_float,
                                            ctypes.c_float)
    cpyquaticus.create_entity.restype = PID



    class USV(ctypes.Structure):
        _fields_ = [
            ("agent_id", ctypes.c_int),
            ("team", ctypes.c_float),
            ("max_speed", ctypes.c_float),
            ("speed_factor", ctypes.c_float),
            ("thrust_map", (ctypes.c_float * 7) * 2),
            ("max_thrust", ctypes.c_float),
            ("max_rudder", ctypes.c_float),
            ("turn_loss", ctypes.c_float),
            ("turn_rate", ctypes.c_float),
            ("max_acc", ctypes.c_float),
            ("max_dec", ctypes.c_float),
            ("dt", ctypes.c_float),
            ("thrust", ctypes.c_float),
            ("prev_pos", ctypes.c_float*2),
            ("pos", ctypes.c_float*2),
            ("speed", ctypes.c_float),
            ("heading", ctypes.c_float),
            ("has_flag", ctypes.c_int),
            ("on_their_side", ctypes.c_int),
            ("tagging_cooldown", ctypes.c_float),
            ("is_tagged", ctypes.c_int),
            ("speed_controller", ctypes.POINTER(PID)),
            ("heading_controller", ctypes.POINTER(PID)),
        ]

    cpyquaticus.create_entity.argtypes = (
        ctypes.c_int,        # agent_id
        ctypes.c_float,      # team
        ctypes.c_float,      # max_speed
        ctypes.c_float,      # speed_factor
        (ctypes.c_float * 7) * 2,  # thrust_map (2D array of floats)
        ctypes.c_float,      # max_thrust
        ctypes.c_float,      # max_rudder
        ctypes.c_float,      # turn_loss
        ctypes.c_float,      # turn_rate
        ctypes.c_float,      # max_acc
        ctypes.c_float,      # max_dec
        ctypes.c_float,      # dt
        ctypes.c_float,      # thrust
        ctypes.POINTER(ctypes.c_float), # prev_pos (pointer to float array)
        ctypes.POINTER(ctypes.c_float), # pos (pointer to float array)
        ctypes.c_float,      # speed
        ctypes.c_float,      # heading
        ctypes.c_int,        # has_flag
        ctypes.c_int,        # on_their_side
        ctypes.c_float,      # tagging_cooldown
        ctypes.c_int,        # is_tagged
        ctypes.POINTER(PID), # speed_controller (pointer to PID structure)
        ctypes.POINTER(PID)  # heading_controller (pointer to PID structure)
    )
    cpyquaticus.create_entity.restype = USV


    cpyquaticus.create_heron.argtypes = (ctypes.c_int,
                                        ctypes.c_int)
    cpyquaticus.create_heron.argtypes = None

    cpyquaticus.move_heron.argtypes = (ctypes.POINTER(USV),
                                        ctypes.c_float,
                                        ctypes.c_float)
    cpyquaticus.move_heron.argtypes = None

    cpyquaticus.free_heron.argtypes = (ctypes.POINTER(USV))
    cpyquaticus.free_heron.argtypes = None

    cpyquaticus.deep_copy_pid.argtypes = (ctypes.POINTER(PID))
    cpyquaticus.deep_copy_pid.argtypes = ctypes.POINTER(PID)

    cpyquaticus.deep_copy_heron.argtypes = (ctypes.POINTER(USV))
    cpyquaticus.deep_copy_heron.argtypes = ctypes.POINTER(USV)

    #End heron.h

    #Start actual_obs.h
    class real_obs(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_float),
            ("y", ctypes.c_float),
            ("heading", ctypes.c_float),
            ("speed", ctypes.c_float),
            ("has_flag", ctypes.c_int),
            ("on_their_side", ctypes.c_int),
            ("tagging_cooldown", ctypes.c_float),
            ("is_tagged", ctypes.c_int),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_float, 
                                            ctypes.c_float, 
                                            ctypes.c_float, 
                                            ctypes.c_float,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_int)
    cpyquaticus.create_entity.restype = team_rewards

    cpyquaticus.create_limit_obs.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int)
    cpyquaticus.create_limit_obs.restype = ctypes.POINTER(real_obs)

    cpyquaticus.generate_obs.argtypes = (ctypes.c_int, ctypes.POINTER(USV))
    cpyquaticus.generate_obs.restype = ctypes.POINTER(real_obs)

    lib.norm_list_obs.argtypes = (ctypes.c_int, 
                                    ctypes.POINTER(real_obs),
                                    ctypes.c_int,
                                    ctypes.POINTER(real_obs),
                                    ctypes.POINTER(real_obs))
    cpyquaticus.norm_list_obs.restype = ctypes.POINTER(ctypes.c_float)

    cpyquaticus.list_obs.argtypes = (ctypes.POINTER(real_obs), 
                                    ctypes.c_int,
                                    ctypes.c_int)
    cpyquaticus.list_obs.restype = ctypes.POINTER(ctypes.c_float)

    cpyquaticus.free_obs.argtypes = (ctypes.POINTER(real_obs))
    cpyquaticus.free_obs.restype = None

    cpyquaticus.free_obs_list.argtypes = (ctypes.POINTER(ctypes.POINTER(real_obs)), ctypes.c_int)
    cpyquaticus.free_obs_list.argtypes = None

    #End actual_obs.h



    #Start Flag.h
    class flag(ctypes.Structure):
        _fields_ = [
            ("team", ctypes.c_int),
            ("pos", ctypes.c_float*2),
            ("grabbed", ctypes.c_int),
            ("grabbed_by", ctypes.c_int),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_int, ctypes.c_float*2, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_entity.restype = flag

    cpyquaticus.create_flag.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_flag.argtypes = ctypes.POINTER(flag)

    cpyquaticus.create_flag_full.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_flag_full.argtypes = ctypes.POINTER(flag)

    cpyquaticus.free_flag.argtypes = (ctypes.POINTER(flag))
    cpyquaticus.free_flag.argtypes = None
    #End Flag.h


    #Start cpyquaticus.c
    class action(ctypes.Structure):
        _fields_ = [
            ("speed", ctypes.c_float),
            ("heading", ctypes.c_float),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_float, ctypes.c_float)
    cpyquaticus.create_entity.restype = action

    class real_obs(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_float),
            ("y", ctypes.c_float),
            ("heading", ctypes.c_float),
            ("speed", ctypes.c_float),
            ("has_flag", ctypes.c_int),
            ("on_their_side", ctypes.c_int),
            ("tagging_cooldown", ctypes.c_float),
            ("is_tagged", ctypes.c_int),
        ]
    cpyquaticus.create_entity.argtypes = (ctypes.c_float, ctypes.c_float)
    cpyquaticus.create_entity.restype = action

    class settings(ctypes.Structure):
        _fields_ = [
            ("field_width", ctypes.c_int),
            ("field_height", ctypes.c_int),
            ("num_agents", ctypes.c_int),
            ("num_teams", ctypes.c_int),
            ("num_steps", ctypes.c_int),
            ("grab_distance", ctypes.c_float),
            ("tagging_distance", ctypes.c_float),
            ("tagging_cooldown_max", ctypes.c_float),
            ("discrete_action_space", ctypes.POINTER(action)),
            ("rewards", ctypes.POINTER(ctypes.POINTER(team_rewards))),
            ("max", ctypes.POINTER(real_obs)),
            ("min", ctypes.POINTER(real_obs)),
            ("flags", ctypes.POINTER(ctypes.POINTER(flag))),
        ]
    # Define the argument types for the create_entity function for settings
    cpyquaticus.create_entity.argtypes = (
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int, 
        ctypes.c_int,  
        ctypes.c_int,  
        ctypes.c_float,  
        ctypes.c_float, 
        ctypes.c_float,  
        ctypes.POINTER(action),  
        ctypes.POINTER(ctypes.POINTER(team_rewards)), 
        ctypes.POINTER(real_obs),  
        ctypes.POINTER(real_obs),  
        ctypes.POINTER(ctypes.POINTER(flag))  
    )
    cpyquaticus.create_entity.restype = settings

    class episode(ctypes.Structure):
        _fields_ = [
            ("agents", ctypes.POINTER(ctypes.POINTER(USV))), 
            ("rewards", ctypes.POINTER(ctypes.POINTER(agent_rewards))),  
        ]
    cpyquaticus.create_entity.argtypes = (
        ctypes.POINTER(ctypes.POINTER(USV)), 
        ctypes.POINTER(ctypes.POINTER(agent_rewards)), 
    )
    cpyquaticus.create_entity.restype = episode

    cpyquaticus.create_game.argtypes = (ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int)
    cpyquaticus.create_game.argtypes = ctypes.POINTER(settings)

    cpyquaticus.free_game.argtypes = (ctypes.POINTER(settings))
    cpyquaticus.free_game.argtypes = None


    cpyquaticus.create_episode.argtypes = (ctypes.POINTER(settings))
    cpyquaticus.create_episode.argtypes = ctypes.POINTER(episode)

    cpyquaticus.create_episode.argtypes = (ctypes.POINTER(episode), ctypes.c_int)
    cpyquaticus.create_episode.argtypes = None

    cpyquaticus.check_game_events.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode))
    cpyquaticus.check_game_events.argtypes = None

    cpyquaticus.step.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode), ctypes.POINTER(actions))
    cpyquaticus.step.argtypes = None

    cpyquaticus.initialize_game_starts.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode))
    cpyquaticus.initialize_game_starts.argtypes = None
    return cpyquaticus