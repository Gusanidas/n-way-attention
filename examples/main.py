from nway_attention.modules.transformer_models import Transformer, Triformer, TriformerCube
from transformer_lens import HookedTransformer
import torch as t
import wandb
from time import time
from nway_attention.train.trainer import Trainer, TransformerTrainingArgs
from nway_attention.cfgs import Config
from nway_attention.train.task_generators import generate_bool_expr, generate_arithmetic_expr, generate_lis, generate_subpal, generate_knapsack, generate_rep
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
filename = "may6lisb.txt"

def train_run(model = "transformer",
 generator = "arithmetic",
  depth=2,
  train_size = 180000,
  test_size = 2000,
   **kwargs):
    model_cfg = Config(
        d_model = kwargs.get("d_model", 128),
        debug = kwargs.get("debug", True),
        layer_norm_eps = kwargs.get("layer_norm_eps", 1e-5),
        d_vocab = kwargs.get("d_vocab", 101),
        init_range = kwargs.get("init_range", 0.02),
        n_ctx = kwargs.get("n_ctx", 42),
        d_head = kwargs.get("d_head", 32),
        d_mlp = kwargs.get("d_mlp", 512),
        n_heads = kwargs.get("n_heads", 4),
        n_layers = kwargs.get("n_layers", 4),
        mlp_type=kwargs.get("mlp_type", "all"),
        with_ln=kwargs.get("with_ln", True),
        order_attn=kwargs.get("order_attn", True),
        attn_eq=kwargs.get("attn_eq", True),
    )

    trainer_args = TransformerTrainingArgs(
        batch_size = kwargs.get("batch_size", 512),
        epochs = kwargs.get("epochs", 100),
        max_steps_per_epoch = kwargs.get("max_steps_per_epoch", 2500),
        lr = kwargs.get("lr", 4e-4),
        weight_decay = kwargs.get("weight_decay", 1e-2),
        wandb_project = kwargs.get("wandb_project", "may6lisb"),
        wandb_name = kwargs.get("wandb_name", None),
        decay_scheduler=kwargs.get("decay_scheduler", "cosine"),
        only_last = kwargs.get("only_last", True),
    )

    if generator == "lis":
        generate_expr = generate_lis
    elif generator == "subpal":
        generate_expr = generate_subpal
    elif generator == "knapsack":
        generate_expr = generate_knapsack
    else:
        raise ValueError("generator must be 'arithmetic', 'boolean', 'lis', 'subpal' or 'knapsack'")
    
    train_list = [generate_expr(device, n=depth) for i in range(train_size)]
    test_list = [generate_expr(device, n=depth) for i in range(test_size)]


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


epochs = 55



size2 = {"d_model": 32*6, "d_head": 32, "d_mlp": 512, "n_heads": 6, "batch_size":512, "lr": 3e-4, "mlp_type": "all", "decay_scheduler": "exponential"}

with open(filename, "a") as file:
    file.write(", ".join(f"{key}: {value}" for key, value in size2.items()) +"\n" + "\n")

generator = "lis"
#n_layers = 10
#
#depth = 14
#train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#
#depth = 21
#train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#
#depth = 27
#train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#
#depth = 37
#train_run(epochs=epochs, model="transformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"trans-{generator}-depth{depth}-layers{n_layers}-size2", **size2)
#
####
size3 = {"d_model": 128, "d_head": 32, "d_mlp": 512, "n_heads": 4, "batch_size":200, "lr": 3e-4, "mlp_type": "all", "decay_scheduler": "exponential"}

with open(filename, "a") as file:
    file.write(", ".join(f"{key}: {value}" for key, value in size2.items()) +"\n" + "\n")

generator = "lis"
n_layers = 4
#
#depth = 14
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size3", **size3)
#
#depth = 21
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size3", **size3)
#
#depth = 27
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size3", **size3)
#
#depth = 37
#train_run(epochs=epochs, model="triformer", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"tri-{generator}-depth{depth}-layers{n_layers}-size3", **size3)
#
#
##########
#
#
#with open(filename, "a") as file:
#    file.write(", ".join(f"{key}: {value}" for key, value in size2.items()) +"\n" + "\n")
#
#generator = "lis"
#n_layers = 4
#
depth = 14
train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size3)

depth = 21
train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size3)

depth = 27
train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size3)

depth = 37
train_run(epochs=epochs, model="triformerCube", generator=generator, n_layers=n_layers, depth=depth, wandb_name=f"triCube-{generator}-depth{depth}-layers{n_layers}-size2", **size3)
