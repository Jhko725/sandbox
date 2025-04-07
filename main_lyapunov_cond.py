from functools import partial
from pathlib import Path
from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import diffrax as dfx
import optax
from dynamical_systems.dataset import TimeSeriesDataset
from dynamical_systems.continuous import solve_ode, AbstractODE
from dynamical_systems.metrics import lyapunov_gr
from dynamical_systems.linalg import gram_schmidt
from dynamics_discovery.preprocessing import split_into_chunks

jax.config.update("jax_enable_x64", True)


class ConditionalTangentODE(AbstractODE):
    ode: AbstractODE

    @property
    def dim(self) -> int:
        return self.ode.dim * (self.ode.dim + 1)

    def rhs(
        self,
        t,
        u: tuple[Float[Array, " dim"], Float[Array, "dim dim"]],
        control_fn: Callable,
    ):
        x, Tx = u
        x_ctrl = control_fn(t)

        def rhs_ode(x_):
            return self.ode.rhs(t, x_, None)

        @partial(jax.vmap, in_axes=(None, -1), out_axes=(None, -1))
        def rhs_jac(x_, Tx_i):
            return jax.jvp(rhs_ode, (x_,), (Tx_i,))

        dx = rhs_ode(x)
        _, dTx = rhs_jac(x_ctrl, Tx)

        return dx, dTx


def conditional_lyapunov_gr(
    ode: AbstractODE,
    u0: Float[Array, " dim"],
    t: Float[Array, " time_perturb"],
    coeffs,
    solver=dfx.Tsit5(),
    rtol=1e-6,
    atol=1e-6,
    **diffeqsolve_kwargs,
) -> tuple[Float[Array, "time_perturb dim"], Float[Array, "time_perturb dim"]]:
    """Evaluate the local Lyapunov exponent by integrating the tangent dynamics
    and performing Gram-Schmidt orthonormalization every time step to prevent the
    the tangent vectors from collapsing onto the largest eigenvector direction."""
    ode_tangent = ConditionalTangentODE(ode)
    u0_tangent = (u0, jnp.identity(ode.dim))
    log_norm_sum = jnp.zeros(ode.dim)

    carry = t[0], u0_tangent, log_norm_sum
    spline = dfx.CubicInterpolation(t, coeffs)

    def _inner(carry, t1):
        t0, u0_tangent_, log_norm_sum0 = carry
        u1, u1_tangent = dfx.diffeqsolve(
            dfx.ODETerm(ode_tangent.rhs),
            solver,
            t0,
            t1,
            None,
            u0_tangent_,
            spline.evaluate,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            **diffeqsolve_kwargs,
        ).ys
        u1_tangent_gr, u1_tangent_norm = gram_schmidt(u1_tangent[0])
        log_norm_sum1 = log_norm_sum0 + jnp.log(u1_tangent_norm)
        carry_new = t1, (u1[0], u1_tangent_gr), log_norm_sum1
        return carry_new, (log_norm_sum1, u1[0])

    _, (log_norm_sums, u_vals) = jax.lax.scan(_inner, carry, t[1:])
    log_norm_sums = jnp.concatenate(
        [jnp.expand_dims(log_norm_sum, 0), log_norm_sums], axis=0
    )
    u_vals = jnp.concatenate([jnp.expand_dims(u0, 0), u_vals], axis=0)
    Dt = jnp.expand_dims(t - t[0], axis=-1)
    return log_norm_sums / Dt, u_vals


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(None, 0, 0, 0))
def solve_neuralode(model, t, u0, coeffs):
    lyapunov, u_pred = conditional_lyapunov_gr(
        model,
        u0,
        t,
        coeffs,
        rtol=1e-4,
        atol=1e-4,
        max_steps=4,
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=4),
    )
    return u_pred, lyapunov[-1]


def loss_mse(
    model,
    t_data: Float[Array, "batch time"],
    u_data: Float[Array, "batch time dim"],
    coeffs_batch,
    u0_data=None,
):
    del u0_data
    u_pred, lyapunov = solve_neuralode(model, t_data, u_data[:, 0], coeffs_batch)
    lyapunov = jnp.mean(lyapunov, axis=0)
    lya_loss = jnp.max(jax.nn.relu(lyapunov))
    return jnp.mean((u_pred - u_data) ** 2) + lya_loss, (lyapunov, lya_loss)


def train_vanilla(
    model,
    t_data,
    u_data,
    u0_data=None,
    loss_fn=loss_mse,
    optimizer_fn: Callable = optax.adabelief,
    lr: float = 1e-3,
    max_epochs: int = 5000,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_config: dict | None = None,
):
    coeffs_batched = eqx.filter_vmap(dfx.backward_hermite_coefficients)(t_data, u_data)
    optimizer = optimizer_fn(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @partial(eqx.filter_value_and_grad, has_aux=True)
    def loss_grad_fn(model, t_batch, u_batch, coeffs, u0_batch):
        return loss_fn(model, t_batch, u_batch, coeffs, u0_batch)

    @eqx.filter_jit
    def make_step(model, t_data, u_data, coeffs, u0_data, opt_state):
        (loss, (lyapunov, lya_loss)), grads = loss_grad_fn(
            model, t_data, u_data, coeffs, u0_data
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, lyapunov, lya_loss

    with wandb.init(
        entity=wandb_entity, project=wandb_project, config=wandb_config
    ) as run:
        loss_history = []
        for epoch in range(max_epochs):
            loss, model, opt_state, lya_mean, lya_loss = make_step(
                model, t_data, u_data, coeffs_batched, u0_data, opt_state
            )

            print(f"{epoch=}, {loss=}")
            for i in range(len(lya_mean)):
                # run.log({f"lambda_{i}": wandb.Histogram(lyapunov[:, i])}, step=epoch)
                run.log({f"lambda_cond_{i}_mean": lya_mean[i]}, step=epoch)
            run.log(
                {
                    "mse": loss - lya_loss,
                    "loss": loss,
                    "epoch": epoch,
                    "lambda_cond_max": jnp.max(lya_mean),
                },
                step=epoch,
            )
            loss_history.append(loss.item())

    return model, jnp.asarray(loss_history)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_dict.pop("wandb")

    model = hydra.utils.instantiate(cfg.model)
    dataset = TimeSeriesDataset.load(cfg.data.loadpath)

    # TODO: refactor the standardization part
    u_train_test = dataset.u
    u_train_mean = jnp.mean(u_train_test[0], axis=0)
    u_train_std = jnp.std(u_train_test[0], axis=0)
    u_train_test_norm = (u_train_test - u_train_mean) / u_train_std

    u_train, u_test = u_train_test_norm
    t_train_batched, _ = split_into_chunks(dataset.t, cfg.preprocessing.batch_length)
    u_train_batched, _ = split_into_chunks(u_train, cfg.preprocessing.batch_length)

    optimizer_fn = hydra.utils.get_method(cfg.training.optimizer_fn)

    model = train_vanilla(
        model,
        t_train_batched,
        u_train_batched,
        optimizer_fn=optimizer_fn,
        lr=cfg.training.lr,
        max_epochs=cfg.training.max_epochs,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        wandb_config=config_dict,
    )
    savedir = Path(cfg.checkpointing.savedir)
    eqx.tree_serialise_leaves(
        savedir
        / f"lorenz_length={cfg.preprocessing.batch_length}_key={cfg.model.key}_lyapunov.eqx",
        model,
    )


if __name__ == "__main__":
    main()
