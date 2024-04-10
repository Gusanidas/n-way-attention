from transformer_models import Transformer, Triformer, TriformerCube
from transformer_lens import HookedTransformer
import torch as t
import wandb
from time import time
from trainer import Trainer, TransformerTrainingArgs
from cfgs import Config
from task_generators import generate_bool_expr, generate_arithmetic_expr, generate_lis, generate_subpal, generate_knapsack
from dotenv import load_dotenv
import os

load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("Logged in to Weights & Biases.")
else:
    print("WANDB_API_KEY not found. Please check your .env file.")



device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
t0 = time()
reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
filename = "results_listri.txt"

def train_run(model = "transformer",
 generator = "arithmetic",
  depth=2,
  train_size = 80000,
  test_size = 1000,
   **kwargs):
    model_cfg = Config(
        d_model = kwargs.get("d_model", 96),
        debug = kwargs.get("debug", True),
        layer_norm_eps = kwargs.get("layer_norm_eps", 1e-5),
        d_vocab = kwargs.get("d_vocab", 50257),
        init_range = kwargs.get("init_range", 0.02),
        n_ctx = kwargs.get("n_ctx", 128),
        d_head = kwargs.get("d_head", 16),
        d_mlp = kwargs.get("d_mlp", 192),
        n_heads = kwargs.get("n_heads", 3),
        n_layers = kwargs.get("n_layers", 2),
        mlp_type=kwargs.get("mlp_type", "all"),
    )

    trainer_args = TransformerTrainingArgs(
        batch_size = kwargs.get("batch_size", 12),
        epochs = kwargs.get("epochs", 100),
        max_steps_per_epoch = kwargs.get("max_steps_per_epoch", 2500),
        lr = kwargs.get("lr", 4e-4),
        weight_decay = kwargs.get("weight_decay", 1e-1),
        wandb_project = kwargs.get("wandb_project", "trisolaris_listri"),
        wandb_name = kwargs.get("wandb_name", None),
        decay_scheduler=kwargs.get("decay_scheduler", "cosine")
    )

    if generator == "arithmetic":
        generate_expr = generate_arithmetic_expr
    elif generator == "boolean":
        generate_expr = generate_bool_expr
    elif generator == "lis":
        generate_expr = generate_lis
    elif generator == "subpal":
        generate_expr = generate_subpal
    elif generator == "knapsack":
        generate_expr = generate_knapsack
    else:
        raise ValueError("generator must be 'arithmetic', 'boolean', 'lis', 'subpal' or 'knapsack'")
    
    train_list = [reference_gpt2.to_tokens(generate_expr(depth=depth)).cpu() for i in range(55000)]
    test_list = [reference_gpt2.to_tokens(generate_expr(depth=depth)).cpu() for i in range(1600)]


    if model == "transformer":
        model = Transformer(model_cfg.to_dict()).to(device)
    elif model == "triformer":
        model = Triformer(model_cfg.to_dict()).to(device)
    elif model == "triformerCube":
        model = TriformerCube(model_cfg.to_dict()).to(device)
    else:
        raise ValueError("model must be 'transformer', 'triformerCube' or 'triformer'")
    print("The model is")
    print(model)
    trainer = Trainer(trainer_args, model, train_list, test_list)
    loss, accuracy = trainer.train()
    print(f"loss = {loss}, accuracy = {accuracy} time = {time() - t0}")
    with open(filename, 'a+') as file:
        file.write(f"name = {kwargs.get('wandb_name', 'default')}, loss = {loss}, accuracy = {accuracy}, time = {time() - t0}\n")
    return model


epochs = 1



size2 = {"d_model": 192, "d_head": 48, "d_mlp": 512, "n_heads": 4, "batch_size":12, "lr": 7e-4, "mlp_type": "none", "decay_scheduler": "exponential"}

with open(filename, "a") as file:
    file.write(", ".join(f"{key}: {value}" for key, value in size2.items()) +"\n" + "\n")

generator = "lis"
n_layers = 2

depth = 10
train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size2", **size2)

depth = 12
train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size2", **size2)

depth = 14
train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size2", **size2)

depth = 16
train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size2", **size2)

depth = 18
train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
