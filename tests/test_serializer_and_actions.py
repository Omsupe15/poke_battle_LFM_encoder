import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from poke_env.battle import Battle, Move, Pokemon, PokemonType, Status
from poke_env.player import SingleBattleOrder, DefaultBattleOrder
from game_encoder.action_space import (
    ACTION_SPACE,
    ACTION_TO_ID,
    parse_action_label,
)
from game_encoder.state_serializer import serialize_battle_state
from game_encoder.recording_player import RecordingPlayer


def create_mock_pokemon(species="pikachu", hp_fraction=1.0, status=None, types=None, moves=None, boosts=None):
    mon = MagicMock(spec=Pokemon)
    mon.species = species
    mon.current_hp_fraction = hp_fraction
    mon.status = status
    mon.types = types or ()
    mon.moves = moves or {}
    mon.boosts = boosts or {}
    return mon


def create_mock_move(move_id="thunderbolt", move_type=None, base_power=90):
    move = MagicMock(spec=Move)
    move.id = move_id
    move.type = move_type
    move.base_power = base_power
    return move


class TestActionSpace:
    def test_action_space_size(self):
        assert len(ACTION_SPACE) == 9
        assert ACTION_SPACE == [
            "MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4",
            "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"
        ]

    def test_parse_move_actions(self):
        m1 = create_mock_move("thunderbolt")
        m2 = create_mock_move("voltswitch")
        m3 = create_mock_move("surf")
        m4 = create_mock_move("agility")

        active = create_mock_pokemon(moves={"thunderbolt": m1, "voltswitch": m2, "surf": m3, "agility": m4})
        battle = MagicMock(spec=Battle)
        battle.active_pokemon = active
        battle.team = {"pikachu": active}

        # Test MOVE_1 through MOVE_4
        order1 = SingleBattleOrder(m1)
        assert parse_action_label(battle, order1) == "MOVE_1"

        order2 = SingleBattleOrder(m2)
        assert parse_action_label(battle, order2) == "MOVE_2"

        order3 = SingleBattleOrder(m3)
        assert parse_action_label(battle, order3) == "MOVE_3"

        order4 = SingleBattleOrder(m4)
        assert parse_action_label(battle, order4) == "MOVE_4"

    def test_parse_switch_actions(self):
        active = create_mock_pokemon("pikachu")
        bench1 = create_mock_pokemon("charizard")
        bench2 = create_mock_pokemon("blastoise")
        bench3 = create_mock_pokemon("venusaur")
        bench4 = create_mock_pokemon("snorlax")
        bench5 = create_mock_pokemon("gengar")

        battle = MagicMock(spec=Battle)
        battle.active_pokemon = active
        battle.team = {
            "pikachu": active,
            "charizard": bench1,
            "blastoise": bench2,
            "venusaur": bench3,
            "snorlax": bench4,
            "gengar": bench5,
        }

        order_sw1 = SingleBattleOrder(bench1)
        assert parse_action_label(battle, order_sw1) == "SWITCH_1"

        order_sw3 = SingleBattleOrder(bench3)
        assert parse_action_label(battle, order_sw3) == "SWITCH_3"

        order_sw5 = SingleBattleOrder(bench5)
        assert parse_action_label(battle, order_sw5) == "SWITCH_5"

    def test_invalid_and_default_orders(self):
        battle = MagicMock(spec=Battle)
        default_order = DefaultBattleOrder()
        assert parse_action_label(battle, default_order) is None


class TestStateSerializer:
    def test_full_state_serialization(self):
        m1 = create_mock_move("thunderbolt", PokemonType.ELECTRIC, 90)
        m2 = create_mock_move("voltswitch", PokemonType.ELECTRIC, 70)

        active = create_mock_pokemon(
            species="zapdos",
            hp_fraction=0.85,
            status=None,
            types=[PokemonType.ELECTRIC, PokemonType.FLYING],
            moves={"thunderbolt": m1, "voltswitch": m2},
            boosts={"spa": 1, "spe": 2},
        )
        opp = create_mock_pokemon(
            species="greattusk",
            hp_fraction=1.0,
            status=None,
            types=[PokemonType.GROUND, PokemonType.FIGHTING],
        )
        bench1 = create_mock_pokemon("kingambit", 1.0)
        bench2 = create_mock_pokemon("gholdengo", 0.5)

        battle = MagicMock(spec=Battle)
        battle.turn = 5
        battle.active_pokemon = active
        battle.opponent_active_pokemon = opp
        battle.team = {
            "zapdos": active,
            "kingambit": bench1,
            "gholdengo": bench2,
        }
        rain_weather = MagicMock()
        rain_weather.name = "RAIN"
        battle.weather = {rain_weather: 1}
        stealth_rock = MagicMock()
        stealth_rock.name = "STEALTH_ROCK"
        battle.fields = {stealth_rock: 1}

        state_str = serialize_battle_state(battle)
        assert "turn:5" in state_str
        assert "active:zapdos hp:0.85 type:ELECTRIC/FLYING" in state_str
        assert "spa:+1,spe:+2" in state_str
        assert "moves:thunderbolt[type:ELECTRIC,bp:90]|voltswitch[type:ELECTRIC,bp:70]|empty|empty" in state_str
        assert "opp:greattusk hp:1.00 type:GROUND/FIGHTING" in state_str
        assert "kingambit:1.00,gholdengo:0.50" in state_str
        assert "field:stealth_rock" in state_str
        assert "weather:rain" in state_str

    def test_empty_or_fainted_state_serialization(self):
        battle = MagicMock(spec=Battle)
        battle.turn = 1
        battle.active_pokemon = None
        battle.opponent_active_pokemon = None
        battle.team = {}
        battle.weather = {}
        battle.fields = {}

        state_str = serialize_battle_state(battle)
        assert "turn:1" in state_str
        assert "active:none hp:0.00 type:none stat:none boosts:none" in state_str
        assert "moves:empty|empty|empty|empty" in state_str
        assert "opp:none hp:0.00 type:none stat:none" in state_str
        assert "bench:fainted:0.00,fainted:0.00,fainted:0.00,fainted:0.00,fainted:0.00" in state_str
        assert "field:none weather:none" in state_str


class TestRecordingPlayer:
    def test_recording_and_jsonl_flush(self):
        m1 = create_mock_move("thunderbolt", PokemonType.ELECTRIC, 90)
        active = create_mock_pokemon(moves={"thunderbolt": m1})
        battle = MagicMock(spec=Battle)
        battle.turn = 2
        battle.active_pokemon = active
        battle.opponent_active_pokemon = None
        battle.team = {"pikachu": active}
        battle.weather = {}
        battle.fields = {}

        player = RecordingPlayer(start_listening=False)

        # Mock super().choose_move
        with patch("poke_env.player.SimpleHeuristicsPlayer.choose_move", return_value=SingleBattleOrder(m1)):
            order = player.choose_move(battle)
            assert order == SingleBattleOrder(m1)

        assert len(player.recorded_trajectories) == 1
        record = player.recorded_trajectories[0]
        assert record["label"] == "MOVE_1"
        assert record["label_id"] == 0
        assert "turn:2" in record["text"]

        # Test flush to disk
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            saved_count = player.save_trajectories(tmp_path)
            assert saved_count == 1
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.loads(f.readline())
                assert loaded["label"] == "MOVE_1"
                assert loaded["label_id"] == 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
