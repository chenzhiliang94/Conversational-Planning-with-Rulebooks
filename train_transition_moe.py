from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
import os
import wandb
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mixture_of_experts import HeirarchicalMoE

# Define the transformation function
def transform_samples_wrapper(start, step, use_residuals=False):
    def transform_samples(batch):
        embeddings = batch['embeddings']
        batched = isinstance(embeddings, list) or (len(embeddings.shape) == 3)
        if not batched:
            embeddings = [embeddings]
        inputs = []
        outputs = []
        for d in embeddings:
            inputs += [d[i-step] for i in range(start + step, len(d), 2)]
            if use_residuals:
                outputs += [d[i] - d[i-step] for i in range(start + step, len(d), 2)]
            else:
                outputs += [d[i] for i in range(start + step, len(d), 2)]
        transformed_samples = {
            'inputs': inputs,
            'outputs': outputs
        }

        return transformed_samples
    return transform_samples

def calculate_mean(batch):
    input_sum = batch['inputs'].sum(dim=0)
    output_sum = batch['outputs'].sum(dim=0)
    return {'input_sum': [input_sum], 'output_sum': [output_sum]}

def sum_of_squared_diff(batch, input_mean, output_mean):
    input_squared_diff_sum = (batch['inputs'] - input_mean).square().sum(dim=0)
    output_squared_diff_sum = (batch['outputs'] - output_mean).square().sum(dim=0)
    return {"input_squared_diff_sum": [input_squared_diff_sum], "output_squared_diff_sum": [output_squared_diff_sum]}

def normalize_dataset(batch, input_mean, input_std, output_mean, output_std):
    inputs = (batch['inputs'] - input_mean) / input_std
    outputs = (batch['outputs'] - output_mean) / output_std
    return {'inputs': inputs, 'outputs': outputs}

def load_dataset(**kwargs):
    # Load dataset
    hf_dataset = load_from_disk(kwargs["dataset"]).with_format("torch")
    hf_dataset = hf_dataset.map(
        transform_samples_wrapper(kwargs['start'], kwargs['step'], use_residuals=kwargs['use_residuals']), 
        remove_columns=hf_dataset.column_names, batched=True, batch_size=2000, 
        num_proc=32, 
        )
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=kwargs['seed'], shuffle=True)
    print(f"length of dataset {len(hf_dataset['train']) + len(hf_dataset['test'])}, with train length {len(hf_dataset['train'])} and test length {len(hf_dataset['test'])}")

    print("Calculating mean and std of inputs and outputs...")
    sums_dataset = hf_dataset["train"].map(
        calculate_mean, 
        remove_columns=hf_dataset["train"].column_names, batched=True, batch_size=10000, 
        # num_proc=32, 
        )
    input_mean = sums_dataset["input_sum"].sum(dim=0) / len(hf_dataset["train"])
    output_mean = sums_dataset["output_sum"].sum(dim=0) / len(hf_dataset["train"])
    squared_diff = hf_dataset["train"].map(
        sum_of_squared_diff, 
        remove_columns=hf_dataset["train"].column_names, batched=True, batch_size=10000, 
        # num_proc=32, 
        fn_kwargs={'input_mean': input_mean, 'output_mean': output_mean})
    input_std = torch.sqrt(squared_diff["input_squared_diff_sum"].sum(dim=0) / len(hf_dataset["train"]))
    output_std = torch.sqrt(squared_diff["output_squared_diff_sum"].sum(dim=0) / len(hf_dataset["train"]))

    normalized_train = hf_dataset["train"].map(normalize_dataset, fn_kwargs={
        'input_mean': input_mean, 'input_std': input_std, 'output_mean': output_mean, 'output_std': output_std
    }, batched=True, batch_size=10000)
    normalized_test = hf_dataset["test"].map(normalize_dataset, fn_kwargs={
        'input_mean': input_mean, 'input_std': input_std, 'output_mean': output_mean, 'output_std': output_std
    }, batched=True, batch_size=10000)

    print("Mean and std calculated.")

    # Convert custom dataset to DataLoader for batching
    train_dataset = DataLoader(
        normalized_train, 
        batch_size=kwargs["batch_size"],
    )
    val_dataset = DataLoader(
        normalized_test, 
        batch_size=8192,
    )

    return train_dataset, val_dataset, input_mean, input_std, output_mean, output_std

class RegressionModel(nn.Module):
    def __init__(self, embedding_size = 1024):
        super(RegressionModel, self).__init__()
        self.add_module("model", HeirarchicalMoE(dim = embedding_size))
        self.input_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(True, dtype=bool), requires_grad=False)

    def set_parameters(self, input_mean, input_std, output_mean, output_std, use_residuals):
        self.input_mean = nn.Parameter(input_mean, requires_grad=False)
        self.input_std = nn.Parameter(input_std, requires_grad=False)
        self.output_mean = nn.Parameter(output_mean, requires_grad=False)
        self.output_std = nn.Parameter(output_std, requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(use_residuals, dtype=bool), requires_grad=False)

    def forward(self, x):
        y = (x - self.input_mean) / self.input_std
        y = self.model(y[:,None])[0][:,0,:]
        y = y * self.output_std + self.output_mean
        if self.use_residuals:
            y = x + y
        return y

def initialize_model(device, **kwargs):
    print(f"Initializing model... on device {device}")

    torch.manual_seed(kwargs["seed"])
    model = HeirarchicalMoE(dim = 1024)

    model.to(device)
    if kwargs['n_gpu'] > 1:
        model = DDP(model, device_ids=[device])

    # model = model.double()
    print(model)
    print("Model initialized.")
    return model

def train_transition_model(**kwargs):
    seed=kwargs["seed"]
    epochs=kwargs["epochs"]
    lr=kwargs["lr"]
    gamma=kwargs["gamma"]
    batch_size=kwargs["batch_size"]
    transition_type=kwargs["transition_type"]
    use_wandb = kwargs.get("use_wandb", False)

    if kwargs['n_gpu'] > 1:
        device = int(os.environ["LOCAL_RANK"])
        if device != 0:
            use_wandb = False
    else:
        device = 0

    outdir = f"{kwargs['out_dir']}/seed_{seed}_batch_{batch_size}/{transition_type}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, input_mean, input_std, output_mean, output_std = load_dataset(**kwargs)
    # torch.save({"input_mean": input_mean, "input_std": input_std, "output_mean": output_mean, "output_std": output_std}, f"{outdir}/{transition_type}_normalization_params.pth")

    model = initialize_model(device = device, **kwargs)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Initialize wandb
    if use_wandb:
        run = wandb.init(project=kwargs["wandb_proj"], name=f"seed_{seed}_{transition_type}", config=kwargs)
        run.save("train_transition_distributed.py")
        run.watch(model, log="all", log_graph=True, criterion=criterion)

    if kwargs["continue_from"] is not None:
        checkpoint = torch.load(kwargs["continue_from"])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict']["model"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        start_epoch = 0

    # Training loop
    min_val_loss = float('inf')
    min_val_loss_epoch = 0
    min_train_loss = float('inf')
    min_train_loss_epoch = 0
    regression_model = RegressionModel(kwargs['embedding_size'])
    regression_model.set_parameters(input_mean, input_std, output_mean, output_std, kwargs['use_residuals'])
    for epoch in tqdm(range(start_epoch, epochs), leave=True):
        train_loss = 0.0
        aux_loss = 0.0
        model.train()
        for batch_no, batch in enumerate(tqdm(train_dataset, leave=True, mininterval=10.0)):
            inputs = batch["inputs"].to(device)
            targets = batch["outputs"].to(device)

            optimizer.zero_grad()   # Zero the gradient buffers

            outputs, curr_aux_loss = model(inputs[:, None]) # Forward pass
            loss = criterion(outputs[:,0,:], targets) # Compute the loss

            (loss + curr_aux_loss).backward() # Backward pass

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step() # Update the weights

            train_loss += loss.item()
            aux_loss += curr_aux_loss.item()

            fractional_epoch = epoch + batch_no / len(train_dataset)
            if batch_no % 100 == 0 and use_wandb and fractional_epoch > 0.1:
                run.log({
                    "Epoch": fractional_epoch, 
                    "Intermediate Training Loss": loss.item(), 
                    "Intermediate Aux Loss": curr_aux_loss.item(), 
                    "Intermediate Total Loss": loss.item() + curr_aux_loss.item()
                    })
            # break
        train_loss /= len(train_dataset)
        aux_loss /= len(train_dataset)

        model.eval()

        with torch.no_grad():
            val_loss = 0.0
            for batch in val_dataset:
                inputs = batch['inputs'].to(device)
                targets = batch['outputs'].to(device)

                outputs, aux_loss = model(inputs[:, None])
                val_loss += criterion(outputs[:,0,:], targets).item()
            val_loss /= len(val_dataset)
            tqdm.write(f'{transition_type.ljust(12)}Epoch {epoch + 1}/{epochs}, lr: {lr_scheduler.get_last_lr()[0]:.3e}, Training loss: {train_loss:.5e}, Validation loss: {val_loss:.5e}')

        if use_wandb:
            # Log metrics to wandb
            run.log({
                f"Epoch": epoch+1, 
                f"Training Loss": train_loss, 
                f"Validation Loss": val_loss, 
                f"Aux Loss": aux_loss,
                "lr": lr_scheduler.get_last_lr()[0]
                })
        if True or epoch > 10:
            if val_loss < min_val_loss:
                min_val_loss_epoch = epoch
                min_val_loss = val_loss
                regression_model.model.load_state_dict(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': regression_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'{outdir}/model_min_val.pth')
            if train_loss < min_train_loss:
                min_train_loss_epoch = epoch
                min_train_loss = train_loss
                # breakpoint()
                # regression_model.model.load_state_dict({f"model.{k}": v for k, v in model.state_dict()}, strict=False)
                regression_model.model.load_state_dict(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': regression_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()
                }
                torch.save(checkpoint, f'{outdir}/model_min_train.pth')
        lr_scheduler.step()

    print("Training complete.")
    print(f"Minimum validation loss: {min_val_loss:.5e} at epoch {min_val_loss_epoch}")
    print(f"Minimum training loss: {min_train_loss:.5e} at epoch {min_train_loss_epoch}")

    # Write to a txt file
    with open(f"{outdir}/results.txt", "w") as file:
        file.write("Training complete.\n")
        file.write(f"Minimum validation loss: {min_val_loss} at epoch {min_val_loss_epoch}\n")
        file.write(f"Minimum training loss: {min_train_loss} at epoch {min_train_loss_epoch}\n")
    regression_model.model.load_state_dict(model.state_dict())
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': regression_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, f'{outdir}/model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed for random number generation", default=0)
    parser.add_argument("--epochs", type=int, help="Number of epochs for training", default=100)
    parser.add_argument("--lr", type=float, help="Learning rate for optimizer", default=0.001)
    parser.add_argument("--gamma", type=float, help="Exponential decay gamma for learning rate scheduler", default=0.9)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2048)
    parser.add_argument("--n_gpu", type=int, help="Number of GPUs to use for training", default=1)
    parser.add_argument("--type_index", type=int, help="Index of the transition type to train", default=-1)
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_proj", type=str, help="Wandb project name", default="lm-sys_transition_moe")
    parser.add_argument("--dataset", type=str, help="dataset location", default="embeddings/lmsys-chat-1m_embeddings_1024")
    parser.add_argument("--use_residuals", action="store_true", help="Use residuals for training")
    parser.add_argument("--out_dir", type=str, help="Output directory for models", default="models/deterministic")
    parser.add_argument("--embedding_size", type=int, help="Embedding size", default=1024)
    parser.add_argument("--continue_from", type=str, help="Continue training from a checkpoint", default=None)
    args = vars(parser.parse_args())

    if args["n_gpu"] > 1:
        dist.init_process_group("nccl")

    print(args)

    processes = []
    start_steps = [(1,1),(0,1),(1,2),(0,2)]
    types = ["llm_human", "human_llm", "llm_llm", "human_human"]
    if args["type_index"] >= 0:
        start_steps = [start_steps[args["type_index"]]]
        types = [types[args["type_index"]]]

    if args["use_wandb"]:
        wandb.setup()
    for start_step, transition_type in zip(start_steps, types):
        args['transition_type'] = transition_type
        args['start'] = start_step[0]
        args['step'] = start_step[1]
        train_transition_model(**args)
    #     p = multiprocessing.Process(target=train_transition_model, kwargs=args)
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    if args["use_wandb"]:
        wandb.finish()
    if args["n_gpu"] > 1:
        dist.destroy_process_group()
