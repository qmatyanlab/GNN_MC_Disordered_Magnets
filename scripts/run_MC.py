import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
from pytorch_lightning import seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from MC.state.state import MCState
from MC.engine.metropolis import MetropolisEngine
from MC.calculators.mc_calculator import MCCalculator

from utils.env_variables import CONFIG_PATH, CONFIG_FILENAME
from utils.mc import setup_logging, load_checkpoint, check_config_consistency

def run_mc(cfg: DictConfig) -> None:
    results_path = Path(cfg.mc.save_results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    log_path = Path("results_and_figs") / "MC" / cfg.name / "results"
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    # --- config consistency check (must run before overwriting config.yaml) ---
    config_yaml_path = results_path / "config.yaml"
    if config_yaml_path.exists():
        check_config_consistency(cfg, config_yaml_path)
        logger.info("Config consistency check passed.")
    config_yaml_path.write_text(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_yaml_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build calculator (loads GNN and optionally MLIP) ---
    calculator = MCCalculator.from_cfg(cfg.calculator, device)
    logger.info(f"Calculator: MCCalculator (structural distortion: {cfg.calculator.structural_distortion.mode})")

    # --- build state and moves ---
    # Resume from checkpoint if one exists (interrupted run), otherwise start fresh.
    checkpoint_path = results_path / "state_checkpoint.pkl"
    if checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        logger.info(f"Resumed MC state from checkpoint: {checkpoint_path}")
    else:
        configuration = hydra.utils.instantiate(cfg.system)
        state = MCState(configuration=configuration)
        logger.info("Starting MC from a fresh configuration.")
    moves = hydra.utils.instantiate(cfg.moves)

    # output = calculator(state.configuration)
    # from utils.mc import test_set_up
    # test_set_up(state, moves, calculator)

    # --- run ---
    engine = MetropolisEngine(
        state=state,
        moves=moves,
        calculator=calculator,
        mc_cfg=cfg.mc,
    )
    engine.run()

    logger.info("MC simulation finished.")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_FILENAME, version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.mc.seed, workers=True)

    cfg = cfg.mc
    run_mc(cfg)


if __name__ == "__main__":
    main()
