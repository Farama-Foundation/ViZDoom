import os 
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from vizdoom import scenarios_path


class VizdoomBasic(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomBasic, self).__init__(
           os.path.join(scenarios_path, "basic.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomCorridor(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomCorridor, self).__init__(
           os.path.join(scenarios_path, "deadly_corridor.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomDeathmatch(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomDeathmatch, self).__init__(
           os.path.join(scenarios_path, "deathmatch.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomDefendCenter(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomDefendCenter, self).__init__(
           os.path.join(scenarios_path, "defend_the_center.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomDefendLine(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomDefendLine, self).__init__(
           os.path.join(scenarios_path, "defend_the_line.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomHealthGathering(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomHealthGathering, self).__init__(
           os.path.join(scenarios_path, "health_gathering.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomHealthGatheringSupreme(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomHealthGatheringSupreme, self).__init__(
           os.path.join(scenarios_path, "health_gathering_supreme.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomMyWayHome(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomMyWayHome, self).__init__(
           os.path.join(scenarios_path, "my_way_home.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomPredictPosition(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomPredictPosition, self).__init__(
           os.path.join(scenarios_path, "predict_position.cfg"), frame_skip, depth, labels, position, health
        )


class VizdoomTakeCover(VizdoomEnv):
    def __init__(
        self, frame_skip=1, depth=False, labels=False, position=False, health=False
    ):
        super(VizdoomTakeCover, self).__init__(
           os.path.join(scenarios_path, "take_cover.cfg"), frame_skip, depth, labels, position, health
        )
