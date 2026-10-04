import argparse
import asyncio
import os
import sys
import uuid
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import Battle
from poke_env.player import BattleOrder

# Import canonical serialization and server manager from game_encoder package
from game_encoder import serialize_battle_state
from game_encoder.server_manager import is_server_running, start_local_showdown

CHECKPOINT_DIR = "./lfm25_encoder_checkpoint"

ACTION_LABELS = [
    "MOVE_1", "MOVE_2", "MOVE_3", "MOVE_4",
    "SWITCH_1", "SWITCH_2", "SWITCH_3", "SWITCH_4", "SWITCH_5"
]
ID_TO_ACTION = {idx: label for idx, label in enumerate(ACTION_LABELS)}


# -------------------------------------------------------------
# 1. Custom Model Architecture
# -------------------------------------------------------------
class LFM25ForSequenceClassification(nn.Module):
    """
    Sequence classification wrapper for LiquidAI/LFM2.5 encoder models.
    Loads the bidirectional hybrid backbone and passes mean-pooled hidden states
    through a linear classification head (d_hidden -> 9 discrete actions).
    """

    def __init__(self, config, num_labels: int = len(ACTION_LABELS)):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # Instantiate encoder from config locally (no remote download required)
        self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs[0]

        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = last_hidden_state.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits)


# -------------------------------------------------------------
# 2. Spectatable Agent with Live Terminal Commentary
# -------------------------------------------------------------
class LFMSpectatorPlayer(Player):
    """
    Autonomous battle agent driven by LFM2.5-Encoder predictions.
    Includes paced delays and room links so human spectators can watch live
    in the Pokémon Showdown browser client.
    """

    def __init__(self, model, tokenizer, turn_delay: float = 1.2, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.turn_delay = turn_delay
        self.verbose = verbose
        self.announced_battles = set()

    def _get_legal_mask(self, battle: Battle) -> torch.Tensor:
        """Computes a boolean mask over the 9 discrete actions based on current game legality."""
        mask = torch.zeros(len(ACTION_LABELS), dtype=torch.bool)
        active = battle.active_pokemon

        # 1. Available moves (indices 0..3)
        if active and active.moves and battle.available_moves:
            avail_move_ids = {m.id for m in battle.available_moves}
            for idx, move_id in enumerate(list(active.moves.keys())[:4]):
                if move_id in avail_move_ids:
                    mask[idx] = True

        # 2. Available switches (indices 4..8)
        if battle.available_switches:
            avail_species = {p.species for p in battle.available_switches}
            bench_mons = [p for p in battle.team.values() if p != active]
            for idx, mon in enumerate(bench_mons[:5]):
                if mon in battle.available_switches or mon.species in avail_species:
                    mask[4 + idx] = True

        # Fallback: if no legal actions detected (e.g. trapped or recharge), allow all
        if not mask.any():
            mask[:] = True
        return mask

    async def _handle_battle_message(self, split_messages):
        """Intercepts battle start to print clickable spectator URLs."""
        await super()._handle_battle_message(split_messages)
        for battle_tag in self.battles.keys():
            if battle_tag not in self.announced_battles:
                self.announced_battles.add(battle_tag)
                print("\n" + "=" * 70)
                print(f"[*] NEW MATCH DETECTED: {battle_tag}")
                print(f"[*] Spectate (Local Client)   : http://localhost:8000/{battle_tag}")
                print(f"[*] Spectate (Web Client URL) : https://play.pokemonshowdown.com/~~localhost:8000/{battle_tag}")
                print("=" * 70 + "\n")

    async def choose_move(self, battle: Battle) -> BattleOrder:
        """
        Async move decision with non-blocking turn pacing so spectators can
        comfortably follow the match in the browser.
        """
        if self.turn_delay > 0:
            await asyncio.sleep(self.turn_delay)

        state_str = serialize_battle_state(battle)
        inputs = self.tokenizer(
            state_str,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].clone()
            legal_mask = self._get_legal_mask(battle)

            # Mask out illegal moves/switches
            logits[~legal_mask] = -1e9
            probabilities = torch.softmax(logits, dim=-1)
            action_id = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[action_id].item()

        action = ID_TO_ACTION[action_id]
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        chosen_order = None
        action_desc = action

        # Resolve MOVE action
        if action.startswith("MOVE_"):
            idx = int(action.split("_")[1]) - 1
            if active and active.moves:
                move_ids = list(active.moves.keys())
                if idx < len(move_ids):
                    target_id = move_ids[idx]
                    for move in battle.available_moves:
                        if move.id == target_id:
                            chosen_order = self.create_order(move)
                            action_desc = f"MOVE: {move.id.upper()}"
                            break

        # Resolve SWITCH action
        elif action.startswith("SWITCH_"):
            idx = int(action.split("_")[1]) - 1
            bench_mons = [p for p in battle.team.values() if p != active]
            if idx < len(bench_mons):
                target_mon = bench_mons[idx]
                for switch in battle.available_switches:
                    if switch == target_mon or switch.species == target_mon.species:
                        chosen_order = self.create_order(switch)
                        action_desc = f"SWITCH TO: {switch.species.capitalize()}"
                        break

        # Fallback if selected action cannot be executed
        if chosen_order is None:
            chosen_order = self.choose_random_move(battle)
            action_desc = f"FALLBACK: {chosen_order.message if hasattr(chosen_order, 'message') else str(chosen_order)}"

        if self.verbose:
            active_str = f"{active.species} ({active.current_hp_fraction*100:.0f}%)" if active else "None"
            opp_str = f"{opponent.species} ({opponent.current_hp_fraction*100:.0f}%)" if opponent else "None"
            print(
                f"  [Turn {battle.turn:02d}] {active_str:20s} vs {opp_str:20s} | "
                f"Action: {action_desc:22s} ({action} {confidence*100:4.1f}%)"
            )

        return chosen_order


# -------------------------------------------------------------
# 3. Model Loading Helper
# -------------------------------------------------------------
def load_lfm_agent(checkpoint_dir: str = CHECKPOINT_DIR):
    """Loads tokenizer, config, and weights from local checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: '{checkpoint_dir}'. "
            "Please ensure the fine-tuned model checkpoint is downloaded."
        )

    print(f"[+] Loading tokenizer and config from {checkpoint_dir}...")
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    print("[+] Initializing LFM2.5 model architecture...")
    model = LFM25ForSequenceClassification(config=config, num_labels=len(ACTION_LABELS))

    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")

    print(f"[+] Loading model weights from {weights_path}...")
    if weights_path.endswith(".bin"):
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)

    model.load_state_dict(state_dict)
    model.eval()
    print("[+] Model loaded successfully and set to eval mode.")
    return model, tokenizer


# -------------------------------------------------------------
# 4. Main Battle Execution Loop
# -------------------------------------------------------------
async def run_spectator_session(
    n_battles: int = 5,
    turn_delay: float = 1.2,
    checkpoint_dir: str = CHECKPOINT_DIR,
    opponent_type: str = "heuristics"
):
    torch.set_num_threads(4)

    # 1. Ensure Pokémon Showdown is running on port 8000
    if not is_server_running(port=8000):
        print("[!] Local Pokémon Showdown server not detected on port 8000.")
        print("[*] Launching local Showdown server automatically...")
        try:
            start_local_showdown(port=8000)
        except Exception as e:
            print(f"[ERROR] Could not start server: {e}")
            print("Please run in a separate terminal: node pokemon-showdown start --no-security")
            sys.exit(1)

    # 2. Load model
    model, tokenizer = load_lfm_agent(checkpoint_dir)

    # 3. Instantiate spectator player with unique alphanumeric account name
    session_id = uuid.uuid4().hex[:4]
    agent_acc = AccountConfiguration(f"LFMAgent{session_id}", None)
    agent = LFMSpectatorPlayer(
        model=model,
        tokenizer=tokenizer,
        turn_delay=turn_delay,
        account_configuration=agent_acc,
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
        max_concurrent_battles=1
    )

    # 4. Instantiate opponent bot with unique username
    opp_acc = AccountConfiguration(f"Opp{opponent_type.capitalize()}{session_id}", None)
    if opponent_type == "heuristics":
        opponent = SimpleHeuristicsPlayer(
            account_configuration=opp_acc,
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
            max_concurrent_battles=1
        )
    elif opponent_type == "max_bp":
        opponent = MaxBasePowerPlayer(
            account_configuration=opp_acc,
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
            max_concurrent_battles=1
        )
    else:
        opponent = RandomPlayer(
            account_configuration=opp_acc,
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
            max_concurrent_battles=1
        )

    print("\n" + "=" * 70)
    print(f"Starting {n_battles} match(es): {agent.username} vs {opponent.username}")
    print(f"Format: gen9randombattle | Turn delay: {turn_delay}s")
    print("When each match starts, click the printed URL to watch in your browser!")
    print("=" * 70 + "\n")

    await agent.battle_against(opponent, n_battles=n_battles)

    print("\n" + "=" * 70)
    print("[SESSION COMPLETE]")
    print(f"  Matches Played    : {agent.n_finished_battles}")
    print(f"  LFM25_Agent Wins  : {agent.n_won_battles}")
    print(f"  Opponent Wins     : {opponent.n_won_battles}")
    win_pct = (agent.n_won_battles / max(1, agent.n_finished_battles)) * 100
    print(f"  LFM Win Rate      : {win_pct:.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Watch Fine-Tuned LFM2.5 Agent Battle Live")
    parser.add_argument("--battles", type=int, default=3, help="Number of battles to run (default: 3)")
    parser.add_argument("--delay", type=float, default=1.2, help="Seconds between turns for spectator pacing (default: 1.2)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_DIR, help="Path to checkpoint folder")
    parser.add_argument("--opponent", type=str, choices=["heuristics", "max_bp", "random"], default="heuristics", help="Opponent archetype")

    args = parser.parse_args()
    asyncio.run(
        run_spectator_session(
            n_battles=args.battles,
            turn_delay=args.delay,
            checkpoint_dir=args.checkpoint,
            opponent_type=args.opponent
        )
    )


if __name__ == "__main__":
    main()