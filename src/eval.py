import hydra
import pyrootutils
from omegaconf import DictConfig


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["requirements.txt"],
    dotenv=True,
    pythonpath=True,
)


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="eval.yaml")
def main(cfg: DictConfig) -> None:

    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.task import evaluate

    # evaluate the model
    evaluate(cfg)


if __name__ == "__main__":
    main()
