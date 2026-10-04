import json
import os
from typing import List, Dict, Any, Optional
from poke_env.player import SimpleHeuristicsPlayer, BattleOrder
from poke_env.battle import Battle

from game_encoder.state_serializer import serialize_battle_state
from game_encoder.action_space import parse_action_label, ACTION_TO_ID


class RecordingPlayer(SimpleHeuristicsPlayer):
    """
    Expert heuristic player that intercepts the battle state on every decision turn
    and records (state_string, action_label, label_id) pairs for behavioral cloning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded_trajectories: List[Dict[str, Any]] = []

    def choose_move(self, battle: Battle) -> BattleOrder:
        """
        Intercepts the battle state, queries the heuristic policy for the order,
        and saves the transition into trajectory memory.
        """
        # 1. State serialization BEFORE the decision is taken
        state_text = serialize_battle_state(battle)

        # 2. Get baseline decision from SimpleHeuristicsPlayer
        chosen_order = super().choose_move(battle)

        # 3. Label resolution into discrete 9-class taxonomy
        action_label = parse_action_label(battle, chosen_order)

        # 4. Save valid discrete transitions
        if action_label in ACTION_TO_ID:
            self.recorded_trajectories.append({
                "text": state_text,
                "label": action_label,
                "label_id": ACTION_TO_ID[action_label]
            })

        return chosen_order

    def save_trajectories(self, file_path: str) -> int:
        """
        Flushes recorded trajectories to disk as a JSONL file.
        Returns the number of transitions written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in self.recorded_trajectories:
                f.write(json.dumps(item) + "\n")
        return len(self.recorded_trajectories)

    def clear_trajectories(self) -> None:
        """Clears the trajectory buffer."""
        self.recorded_trajectories.clear()
