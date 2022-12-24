from typing import Optional

import hydra
import pyrootutils
from omegaconf import DictConfig


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(version_base="1.3",
            config_path="../configs",
            config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.task import train
    from src.utils import get_metric_value

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyper-parmeter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
