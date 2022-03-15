import os 
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from vizdoom import scenarios_path


class VizdoomBasic(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomBasic, self).__init__(
           os.path.join(scenarios_path, "basic.cfg"), frame_skip
        )


class VizdoomCorridor(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomCorridor, self).__init__(
           os.path.join(scenarios_path, "deadly_corridor.cfg"), frame_skip
        )


class VizdoomDeathmatch(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomDeathmatch, self).__init__(
           os.path.join(scenarios_path, "deathmatch.cfg"), frame_skip
        )


class VizdoomDefendCenter(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomDefendCenter, self).__init__(
           os.path.join(scenarios_path, "defend_the_center.cfg"), frame_skip
        )


class VizdoomDefendLine(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomDefendLine, self).__init__(
           os.path.join(scenarios_path, "defend_the_line.cfg"), frame_skip
        )


class VizdoomHealthGathering(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomHealthGathering, self).__init__(
           os.path.join(scenarios_path, "health_gathering.cfg"), frame_skip
        )


class VizdoomHealthGatheringSupreme(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomHealthGatheringSupreme, self).__init__(
           os.path.join(scenarios_path, "health_gathering_supreme.cfg"), frame_skip
        )


class VizdoomMyWayHome(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomMyWayHome, self).__init__(
           os.path.join(scenarios_path, "my_way_home.cfg"), frame_skip
        )


class VizdoomPredictPosition(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomPredictPosition, self).__init__(
           os.path.join(scenarios_path, "predict_position.cfg"), frame_skip
        )


class VizdoomTakeCover(VizdoomEnv):
    def __init__(
        self, frame_skip=1
    ):
        super(VizdoomTakeCover, self).__init__(
           os.path.join(scenarios_path, "take_cover.cfg"), frame_skip
        )
