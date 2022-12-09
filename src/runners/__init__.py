REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .ippo_episode_runner import IPPOEpisodeRunner
REGISTRY["ippo"] = IPPOEpisodeRunner

# from .ippo_iql_episode_runner import IPPOIQLEpisodeRunner
# REGISTRY["ippo_iql"] = IPPOIQLEpisodeRunner
