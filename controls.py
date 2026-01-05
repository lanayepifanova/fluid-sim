from dataclasses import dataclass


@dataclass
class ControlState:
    visualization_mode: str
    colormap: str
    paused: bool = False
    show_hands: bool = True
    show_info: bool = True


def get_instructions():
    return [
        "Controls:",
        "1-5: Change visualization mode",
        "C: Cycle colormap",
        "H: Toggle hand visualization",
        "I: Toggle info",
        "SPACE: Pause/Resume",
        "R: Reset",
        "Q: Quit",
    ]


def handle_key(key, state: ControlState, colormaps, fluid_sim):
    if key == ord("q"):
        return "quit"
    if key == ord("1"):
        state.visualization_mode = "density"
        return None
    if key == ord("2"):
        state.visualization_mode = "velocity"
        return None
    if key == ord("3"):
        state.visualization_mode = "particles"
        return None
    if key == ord("4"):
        state.visualization_mode = "streamlines"
        return None
    if key == ord("5"):
        state.visualization_mode = "combined"
        return None
    if key == ord("c"):
        idx = colormaps.index(state.colormap)
        state.colormap = colormaps[(idx + 1) % len(colormaps)]
        return None
    if key == ord("h"):
        state.show_hands = not state.show_hands
        return None
    if key == ord("i"):
        state.show_info = not state.show_info
        return None
    if key == ord(" "):
        state.paused = not state.paused
        return None
    if key == ord("r"):
        fluid_sim.reset()
        return None
    return None
