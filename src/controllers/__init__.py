REGISTRY = {}

from .basic_controller import BasicMAC
from .dcntrl_controller import DcntrlMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dcntrl_mac"] = DcntrlMAC
