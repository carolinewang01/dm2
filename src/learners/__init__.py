from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .iq_learner import IQLearner
from .ippo_learner import IPPOLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["iq_learner"] = IQLearner
REGISTRY["ippo_learner"] = IPPOLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
