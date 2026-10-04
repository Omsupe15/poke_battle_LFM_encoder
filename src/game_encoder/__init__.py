"""
game-encoder: High-throughput Pokémon Showdown data collection and encoder policy toolkit.
"""

from game_encoder.action_space import ACTION_SPACE, ACTION_TO_ID, ID_TO_ACTION, parse_action_label
from game_encoder.state_serializer import serialize_battle_state
from game_encoder.recording_player import RecordingPlayer
from game_encoder.collector import collect_dataset
from game_encoder.server_manager import is_server_running, start_local_showdown

__all__ = [
    "ACTION_SPACE",
    "ACTION_TO_ID",
    "ID_TO_ACTION",
    "parse_action_label",
    "serialize_battle_state",
    "RecordingPlayer",
    "collect_dataset",
    "is_server_running",
    "start_local_showdown",
]


def main() -> None:
    from game_encoder.collector import main as collector_main
    collector_main()
