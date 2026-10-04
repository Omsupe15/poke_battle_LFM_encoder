from typing import Optional, List, Dict
from poke_env.battle import Battle, Move, Pokemon
from poke_env.player import BattleOrder, SingleBattleOrder

ACTION_SPACE: List[str] = [
    "MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4",
    "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"
]

ACTION_TO_ID: Dict[str, int] = {action: i for i, action in enumerate(ACTION_SPACE)}
ID_TO_ACTION: Dict[int, str] = {i: action for i, action in enumerate(ACTION_SPACE)}


def parse_action_label(battle: Battle, order: BattleOrder) -> Optional[str]:
    """
    Normalizes a poke-env BattleOrder into one of 9 discrete macro-action classes:
      - MOVE_1 .. MOVE_4: Corresponds to active Pokémon move slots 0..3
      - SWITCH_1 .. SWITCH_5: Corresponds to bench Pokémon slots 0..4
    Returns None if the order cannot be resolved or is a default/forfeit order.
    """
    if not isinstance(order, SingleBattleOrder):
        return None

    # 1. Move action resolution
    if isinstance(order.order, Move) or hasattr(order.order, "id"):
        move_id = order.order.id
        if battle.active_pokemon and battle.active_pokemon.moves:
            for idx, m_id in enumerate(list(battle.active_pokemon.moves.keys())[:4]):
                if m_id == move_id:
                    return f"MOVE_{idx + 1}"

    # 2. Switch action resolution
    if isinstance(order.order, Pokemon) or hasattr(order.order, "species"):
        target_species = order.order.species
        bench_mons = [
            mon for mon in battle.team.values()
            if mon != battle.active_pokemon
        ]
        for idx, mon in enumerate(bench_mons[:5]):
            if mon.species == target_species:
                return f"SWITCH_{idx + 1}"

    return None
