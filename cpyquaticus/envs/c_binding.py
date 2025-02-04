import ctypes
import numpy as np
from structures import team_rewards, agent_rewards, PID, USV, real_obs, flag, action, settings, episode


# Loads in the the c methods
def load_cpyquaticus(c_file=None):
    try:
        if c_file == None or c_file == 'linux':
            cpyquaticus = ctypes.CDLL("./cpyquaticus.so")
        else:
            cpyquaticus = ctypes.CDLL("./cpyquaticus.dylib")
    except OSError as e:
        raise RuntimeError(f"Failed to load shared library: {e}")


    #Start reward.h

    cpyquaticus.create_team_reward.argtypes = None
    cpyquaticus.create_team_reward.restype = ctypes.POINTER(team_rewards)

    cpyquaticus.free_team_reward.argtypes = (ctypes.POINTER(team_rewards),)
    cpyquaticus.free_team_reward.restype = None

    cpyquaticus.free_agent_reward.argtypes = (ctypes.POINTER(agent_rewards),)
    cpyquaticus.free_agent_reward.restype = None

    # End rewards.h

    # Start heron.h

    cpyquaticus.create_heron.argtypes = (ctypes.c_int,
                                        ctypes.c_int)
    cpyquaticus.create_heron.restype = None

    cpyquaticus.move_heron.argtypes = (ctypes.POINTER(USV),
                                        ctypes.c_float,
                                        ctypes.c_float)
    cpyquaticus.move_heron.restype = None

    cpyquaticus.free_heron.argtypes = (ctypes.POINTER(USV),)
    cpyquaticus.free_heron.restype = None

    cpyquaticus.deep_copy_pid.argtypes = (ctypes.POINTER(PID),)
    cpyquaticus.deep_copy_pid.restype = ctypes.POINTER(PID)

    cpyquaticus.deep_copy_heron.argtypes = (ctypes.POINTER(USV),)
    cpyquaticus.deep_copy_heron.restype = ctypes.POINTER(USV)

    # #End heron.h

    # #Start actual_obs.h

    cpyquaticus.create_limit_obs.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int)
    cpyquaticus.create_limit_obs.restype = ctypes.POINTER(real_obs)

    cpyquaticus.generate_obs.argtypes = (ctypes.c_int, ctypes.POINTER(USV))
    cpyquaticus.generate_obs.restype = ctypes.POINTER(real_obs)

    cpyquaticus.norm_list_obs.argtypes = (ctypes.c_int, 
                                    ctypes.POINTER(real_obs),
                                    ctypes.c_int,
                                    ctypes.POINTER(real_obs),
                                    ctypes.POINTER(real_obs))
    cpyquaticus.norm_list_obs.restype = ctypes.POINTER(ctypes.c_float)

    cpyquaticus.list_obs.argtypes = (ctypes.POINTER(real_obs), 
                                    ctypes.c_int,
                                    ctypes.c_int)
    cpyquaticus.list_obs.restype = ctypes.POINTER(ctypes.c_float)

    cpyquaticus.free_obs.argtypes = (ctypes.POINTER(real_obs),)
    cpyquaticus.free_obs.restype = None

    cpyquaticus.free_obs_struct.argtypes = (ctypes.POINTER(ctypes.POINTER(real_obs)), ctypes.c_int)
    cpyquaticus.free_obs_struct.restype = None

    cpyquaticus.free_obs_list.argtypes = (ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int)
    cpyquaticus.free_obs_list.restype = None

    # #End actual_obs.h

    cpyquaticus.create_flag.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_flag.restype = ctypes.POINTER(flag)

    cpyquaticus.create_flag_full.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cpyquaticus.create_flag_full.restype = ctypes.POINTER(flag)

    cpyquaticus.free_flag.argtypes = (ctypes.POINTER(flag),)
    cpyquaticus.free_flag.restype = None

    #End Flag.h


    #Start cpyquaticus.c

    cpyquaticus.create_game.argtypes = (ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int)
    cpyquaticus.create_game.restype = ctypes.POINTER(settings)

    cpyquaticus.free_game.argtypes = (ctypes.POINTER(settings),)
    cpyquaticus.free_game.restype = None


    cpyquaticus.create_episode.argtypes = (ctypes.POINTER(settings),)
    cpyquaticus.create_episode.restype = ctypes.POINTER(episode)

    cpyquaticus.free_episode.argtypes = (ctypes.POINTER(episode), ctypes.c_int)
    cpyquaticus.free_episode.restype = None

    cpyquaticus.check_game_events.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode))
    cpyquaticus.check_game_events.restype = None

    cpyquaticus.step.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode), ctypes.POINTER(ctypes.c_int))
    cpyquaticus.step.restype = None

    cpyquaticus.initialize_game_starts.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode))
    cpyquaticus.initialize_game_starts.restype = None

    cpyquaticus.get_observations.argtypes = (ctypes.POINTER(settings), ctypes.POINTER(episode), ctypes.c_int, ctypes.POINTER(real_obs), ctypes.POINTER(real_obs))
    cpyquaticus.get_observations.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

    #End cpyquaticus.c
    return cpyquaticus
