import numpy as np
from scipy.optimize import fmin_slsqp, basinhopping


class Solver:
    def estimate(
        self,
        treated_predictors,
        control_predictors,
        treated_outcome,
        control_outcome,
        control_outcome_after,
    ):
        num_predictors, num_control_units = (
            control_predictors.shape[0],
            control_predictors.shape[1],
        )
        v0 = [1.0 / num_predictors] * num_predictors
        w0 = np.array([1.0 / num_control_units] * num_control_units)
        predictors_importance = self._get_v_star(
            v0,
            w0,
            treated_predictors,
            control_predictors,
            treated_outcome,
            control_outcome,
        )
        control_unit_weights = self._get_weights_star(
            w0, predictors_importance, treated_predictors, control_predictors
        )

        treated_outcome_estimate = control_outcome_after @ control_unit_weights

        return (
            treated_outcome_estimate,
            predictors_importance,
            control_unit_weights,
        )

    def _weights_target_func(
        self, weights, v, treated_predictors, control_predictors
    ):
        importance_matrix = np.diag(v)
        weights_func_value = (
            (treated_predictors - (control_predictors @ weights)).transpose()
            @ importance_matrix
            @ (treated_predictors - (control_predictors @ weights))
        )
        return weights_func_value.item(0)

    def _weights_constraint(
        self, weights, v, treated_predictors, control_predictors
    ):
        return np.sum(weights) - 1

    def _v_target_func(self, weights, treated_outcome, control_outcome):
        v_func_value = (
            treated_outcome - (control_outcome @ weights)
        ).transpose() @ (treated_outcome - (control_outcome @ weights))
        return v_func_value.item(0)

    def _get_v_loss(
        self,
        v0,
        weights0,
        treated_predictors,
        control_predictors,
        treated_outcome,
        control_outcome,
    ):
        # We will be passing the function different values of importance (V)
        # as we call this function later and we want to get the optimal
        # values of the weights given those values
        weights = fmin_slsqp(
            self._weights_target_func,
            weights0,
            f_eqcons=self._weights_constraint,
            bounds=[(0.0, 1.0)] * len(weights0),
            args=(v0, treated_predictors, control_predictors),
            disp=False,
        )
        loss = self._v_target_func(weights, treated_outcome, control_outcome)
        return loss

    def _get_v_star(
        self,
        v0,
        weights0,
        treated_predictors,
        control_predictors,
        treated_outcome,
        control_outcome,
    ):
        result = basinhopping(
            self._get_v_loss,
            v0,
            niter=10,
            seed=2021,
            minimizer_kwargs=dict(
                method="L-BFGS-B",
                args=(
                    weights0,
                    treated_predictors,
                    control_predictors,
                    treated_outcome,
                    control_outcome,
                ),
                bounds=[(0.0, 1.0)] * len(v0),
            ),
        )
        v_star = result["x"]
        return v_star

    def _get_weights_star(
        self, weights0, v_star, treated_predictors, control_predictors
    ):
        weights_star = fmin_slsqp(
            self._weights_target_func,
            weights0,
            f_eqcons=self._weights_constraint,
            bounds=[(0.0, 1.0)] * len(weights0),
            args=(v_star, treated_predictors, control_predictors),
            disp=False,
        )

        return weights_star
