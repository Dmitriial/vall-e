import logging
from collections import defaultdict
from typing import Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm

from vall_e.utils.engines import Stats
from .config import get_cfg
from .data import create_train_val_dataloader
from .emb import qnt
from .utils import setup_logging, to_device, trainer
from .vall_e import get_model

# output dynamic data
_logger = logging.getLogger(__name__)


def load_engines():
    model = get_model(get_cfg().model)

    engines = dict(
        model=trainer.Engine(
            model=model,
            config=get_cfg().ds_cfg,
        ),
    )

    return trainer.load_engines(engines, get_cfg())


def main():
    setup_logging(get_cfg().log_dir)

    train_dl, subtrain_dl, val_dl = create_train_val_dataloader()

    def train_feeder(engines, batch, name) -> Tuple[Tensor, Stats]:
        model = engines["model"]

        if get_cfg().model.startswith("ar"):
            _ = model(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resp_list=batch["resp"],
            )
        elif get_cfg().model.startswith("nar"):
            _ = model(
                text_list=batch["text"],
                proms_list=batch["proms"],
                resps_list=batch["resps"],
            )
        else:
            raise NotImplementedError(get_cfg().model)

        losses = model.gather_attribute("loss")

        loss = torch.stack([*losses.values()]).sum()

        stats = {}
        stats |= {k: v.item() for k, v in losses.items()}
        stats |= engines.gather_attribute("scalar")

        return loss, stats

    @torch.inference_mode()
    def run_eval(engines, name, dl) -> Stats:
        model = engines["model"]
        stats = defaultdict(list)

        aggregated_losses = []
        for batch in tqdm(dl):
            batch: dict = to_device(batch, get_cfg().device)

            if get_cfg().model.startswith("ar"):
                resp_list = model(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    resp_list=batch["resp"],
                    max_steps=get_cfg().max_val_ar_steps,
                    sampling_temperature=get_cfg().sampling_temperature,
                )
                _ = [r.unsqueeze(-1) for r in resp_list]
            elif get_cfg().model.startswith("nar"):
                _ = model(
                    text_list=batch["text"],
                    proms_list=batch["proms"],
                    resps_list=batch["resps"],
                    sampling_temperature=get_cfg().sampling_temperature,
                )
            else:
                raise NotImplementedError(get_cfg().model)

            losses = model.gather_attribute("loss")
            loss = sum([v.item() for k, v in losses.items()])
            assert loss != 0, "It's a magic case. Exact accuracy!"
            aggregated_losses.append(loss)

        qnt.unload_model()

        stats = {k: sum(v) / len(v) for k, v in stats.items()}
        stats["global_step"] = engines.global_step
        stats["name"] = name
        stats["loss"] = sum(aggregated_losses)

        return stats

    def eval_fn(engines):
        stats_val = run_eval(engines, "val", val_dl)

        return {'val_loss': stats_val['loss']}

    trainer.train(
        engines_loader=load_engines,
        train_dl=train_dl,
        train_feeder=train_feeder,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    main()
