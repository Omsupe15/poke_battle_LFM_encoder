from typing import List
from poke_env.battle import Battle


def serialize_battle_state(battle: Battle) -> str:
    """
    Serializes a poke-env Battle state into an order-consistent, token-efficient
    domain-specific string representation suitable for bidirectional encoders.
    """
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon

    # 1. Active Pokémon features
    if active:
        active_name = active.species or "none"
        hp_frac = active.current_hp_fraction if active.current_hp_fraction is not None else 0.0
        active_hp = f"{hp_frac:.2f}"
        active_status = active.status.name.lower() if (active.status and hasattr(active.status, "name")) else (str(active.status).lower() if active.status else "none")

        if active.types:
            type_names = [
                t.name if hasattr(t, "name") else str(t)
                for t in active.types
                if t
            ]
            active_types = "/".join(type_names) if type_names else "none"
        else:
            active_types = "none"

        boost_items = [
            f"{stat}:{boost:+d}"
            for stat, boost in sorted(active.boosts.items())
            if boost != 0
        ]
        active_boosts = ",".join(boost_items) if boost_items else "none"
    else:
        active_name, active_hp, active_status, active_types, active_boosts = (
            "none", "0.00", "none", "none", "none"
        )

    # 2. Opponent Active Pokémon features
    if opponent:
        opp_name = opponent.species or "none"
        opp_hp_frac = opponent.current_hp_fraction if opponent.current_hp_fraction is not None else 0.0
        opp_hp = f"{opp_hp_frac:.2f}"
        opp_status = opponent.status.name.lower() if (opponent.status and hasattr(opponent.status, "name")) else (str(opponent.status).lower() if opponent.status else "none")
        if opponent.types:
            opp_type_names = [
                t.name if hasattr(t, "name") else str(t)
                for t in opponent.types
                if t
            ]
            opp_types = "/".join(opp_type_names) if opp_type_names else "none"
        else:
            opp_types = "none"
    else:
        opp_name, opp_hp, opp_status, opp_types = "none", "0.00", "none", "none"

    # 3. Available Moves (Padded to 4 slots)
    move_tokens: List[str] = []
    if active and active.moves:
        for m_id, move in list(active.moves.items())[:4]:
            move_type = move.type.name if (move.type and hasattr(move.type, "name")) else (str(move.type) if move.type else "NORMAL")
            bp = move.base_power or 0
            move_tokens.append(f"{m_id}[type:{move_type},bp:{bp}]")
    while len(move_tokens) < 4:
        move_tokens.append("empty")
    moves_str = "|".join(move_tokens)

    # 4. Bench Team Health Summary (Padded to 5 slots)
    bench_tokens: List[str] = []
    if battle.team:
        for mon in battle.team.values():
            if mon != active:
                m_hp = mon.current_hp_fraction if mon.current_hp_fraction is not None else 0.0
                m_name = mon.species or "unknown"
                bench_tokens.append(f"{m_name}:{m_hp:.2f}")
    while len(bench_tokens) < 5:
        bench_tokens.append("fainted:0.00")
    bench_str = ",".join(bench_tokens[:5])

    # 5. Global Field & Weather Conditions
    weather = "none"
    if battle.weather:
        weather_key = next(iter(battle.weather.keys()), None)
        if weather_key:
            weather = weather_key.name.lower() if hasattr(weather_key, "name") else str(weather_key).lower()

    fields = "none"
    if battle.fields:
        field_names = [
            f.name.lower() if hasattr(f, "name") else str(f).lower()
            for f in battle.fields.keys()
        ]
        if field_names:
            fields = ",".join(sorted(field_names))

    turn_num = battle.turn if battle.turn is not None else 0

    return (
        f"turn:{turn_num} | "
        f"active:{active_name} hp:{active_hp} type:{active_types} stat:{active_status} boosts:{active_boosts} | "
        f"moves:{moves_str} | "
        f"opp:{opp_name} hp:{opp_hp} type:{opp_types} stat:{opp_status} | "
        f"bench:{bench_str} | "
        f"field:{fields} weather:{weather}"
    )
