"""
Hyperparameter tuning for BLIP using Optuna
"""

import itertools

import optuna
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from blip import BLIP
from dataloader import get_dataloader
from train import compute_losses


def objective(trial):
    """Optuna objective function for hyperparameter search"""

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    # Fixed parameters
    batch_size = 16

    # Training settings
    max_steps = 10000  # Sufficient steps for convergence
    val_interval = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = BLIP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="train"
    )
    val_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="validation"
    )
    train_iter = itertools.islice(train_loader, max_steps)

    model.train()
    best_val_loss = float("inf")

    for i, batch in enumerate(train_iter):
        if batch is None:
            continue

        images, input_ids, attention_mask = batch
        images, input_ids, attention_mask = (
            images.to(device),
            input_ids.to(device),
            attention_mask.to(device),
        )

        # Compute losses
        loss_itc, loss_itm, loss_lm, _ = compute_losses(
            model, images, input_ids, attention_mask, device
        )

        # Weighted loss combination
        loss = loss_itc + loss_itm + loss_lm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation and pruning
        if i > 0 and i % val_interval == 0:
            model.eval()
            val_loss = 0
            val_loss_itc = 0
            val_loss_itm = 0
            val_loss_lm = 0

            with torch.no_grad():
                val_iter = itertools.islice(val_loader, 50)
                for val_batch in val_iter:
                    if val_batch is None:
                        continue

                    val_images, val_input_ids, val_attention_mask = val_batch
                    val_images, val_input_ids, val_attention_mask = (
                        val_images.to(device),
                        val_input_ids.to(device),
                        val_attention_mask.to(device),
                    )

                    v_loss_itc, v_loss_itm, v_loss_lm, _ = compute_losses(
                        model, val_images, val_input_ids, val_attention_mask, device
                    )

                    val_loss_itc += v_loss_itc.item()
                    val_loss_itm += v_loss_itm.item()
                    val_loss_lm += v_loss_lm.item()
                    val_loss += (v_loss_itc + v_loss_itm + v_loss_lm).item()

            avg_val_loss = val_loss / 50
            avg_val_loss_itc = val_loss_itc / 50
            avg_val_loss_itm = val_loss_itm / 50
            avg_val_loss_lm = val_loss_lm / 50

            model.train()

            # Report intermediate values for pruning
            trial.report(avg_val_loss, i)

            # Update best loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            print(
                f"Trial {trial.number} | Step {i} | Val Loss: {avg_val_loss:.4f} "
                f"(ITC: {avg_val_loss_itc:.4f}, ITM: {avg_val_loss_itm:.4f}, LM: {avg_val_loss_lm:.4f})"
            )

    return best_val_loss


def tune_hyperparameters(n_trials=20, storage=None):
    """Run hyperparameter tuning with Optuna

    Args:
        n_trials: Number of trials to run
        storage: Optuna storage URL (e.g., 'sqlite:///optuna.db' for persistence)
    """

    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name="blip_hyperparameter_tuning",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=2000,  # Don't prune before step 2000
        ),
    )

    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    print("=" * 70)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Print results
    print("\n" + "=" * 70)
    print("Hyperparameter Tuning Complete!")
    print("=" * 70)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Print top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value")
    print(
        trials_df[["number", "value", "params_learning_rate", "params_weight_decay"]]
        .head(5)
        .to_string()
    )

    # Save study
    if storage is None:
        # Save to default location
        import joblib

        joblib.dump(study, "optuna_study.pkl")
        print("\nStudy saved to: optuna_study.pkl")

    return study


if __name__ == "__main__":
    # Run tuning
    # Use SQLite for persistence (optional)
    storage = "sqlite:///blip_optuna.db"

    study = tune_hyperparameters(n_trials=20, storage=storage)

    # You can also visualize results (requires optuna-dashboard or plotly)
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()
