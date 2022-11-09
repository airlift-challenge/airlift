from typing import List

PlaneTypeID = int


class PlaneType:
    """
    Manages Airplane Models and characterisitics common to all airplanes of a given type.
    These can also be directly passing to the world generator when creating the environment.
    An example of generating three airplane types:
    EX: AirliftWorldGenerator(
            plane_types=[PlaneType(id=0, model='A0', max_range=.40, speed=.1, max_weight=5),
                         PlaneType(id=1, model='A1', max_range=.41, speed=.1, max_weight=20),
                         PlaneType(id=2, model='A2', max_range=.76, speed=.3, max_weight=2)],
    """

    def __init__(self, id: PlaneTypeID, max_range: float = 1, speed: float = 1, max_weight: float = 1,
                 model: str = None):
        self.id = id

        if model is None:
            self.model = 'A{}'.format(id)
        else:
            self.model = model

        self.max_range = max_range
        self.speed = speed
        self.max_weight = max_weight

    def __str__(self):
        return self.model

    def __repr__(self):
        return self.model
