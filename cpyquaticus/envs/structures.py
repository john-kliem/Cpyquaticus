import ctypes
import numpy as np

## All The Utility Structures Used in C Pyquaticus Implementation ##

#Start reward.h
class team_rewards(ctypes.Structure):
    _fields_ = [
        ("team_grab", ctypes.c_int),
        ("team_capture", ctypes.c_int),
        ("opp_grab", ctypes.c_int),
        ("opp_capture", ctypes.c_int),
    ]
# cpyquaticus.create_team_rewards.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int)
# cpyquaticus.create_team_rewards.restype = team_rewards
class agent_rewards(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_int),
        ("grab", ctypes.c_int),
        ("capture", ctypes.c_int),
        ("got_tagged", ctypes.c_int),
        ("oob", ctypes.c_int),
    ]
#End reward.h

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
#End actual_obs.h

#Start flag.h
class flag(ctypes.Structure):
    _fields_ = [
        ("team", ctypes.c_int),
        ("pos", ctypes.c_float*2),
        ("grabbed", ctypes.c_int),
        ("grabbed_by", ctypes.c_int),
    ]
#End flag.h

#Start cpyquaticus.c structs
class action(ctypes.Structure):
    _fields_ = [
        ("speed", ctypes.c_float),
        ("heading", ctypes.c_float),
    ]

class settings(ctypes.Structure):
    _fields_ = [
        ("field_width", ctypes.c_int),
        ("field_height", ctypes.c_int),
        ("num_agents", ctypes.c_int),
        ("num_teams", ctypes.c_int),
        ("num_steps", ctypes.c_int),
        ("speed_up_factor", ctypes.c_int),
        ("grab_distance", ctypes.c_float),
        ("tagging_distance", ctypes.c_float),
        ("tagging_cooldown_max", ctypes.c_float),
        ("discrete_action_space", ctypes.POINTER(action)),
        ("rewards", ctypes.POINTER(ctypes.POINTER(team_rewards))),
        ("max", ctypes.POINTER(real_obs)),
        ("min", ctypes.POINTER(real_obs)),
        ("flags", ctypes.POINTER(ctypes.POINTER(flag))),
    ]
class episode(ctypes.Structure):
    _fields_ = [
        ("agents", ctypes.POINTER(ctypes.POINTER(USV))), 
        ("rewards", ctypes.POINTER(ctypes.POINTER(agent_rewards))),  
    ]
#End cpyquaticus.c structs

## End of required structures