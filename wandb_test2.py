import wandb
import yaml


def train(sweep_id):
    default_config = {"a": 1, "b": 2}

    for fold in range(5):
        wandb.init(
            project="Image_Classification",
            group="sweep_test",
            job_type=sweep_id,
            reinit=True,
            config=default_config,
            name=f"sweep_test / {sweep_id} / Fold_{fold}",
            resume="allow",
        )
        a = wandb.config.a
        b = wandb.config.b

        for i in range(10):
            a += 1
            b += 2
            wandb.log({"test": a, "test2": b})

        wandb.join()


if __name__ == "__main__":

    with open("wandb_test.yaml") as f:
        sweep_config = yaml.safe_load(f)  # sweep config.yaml
    sweep_id = wandb.sweep(sweep_config, project="Image_Classification")

    wandb.agent(
        sweep_id, function=train(sweep_id=sweep_id), count=10,
    )
