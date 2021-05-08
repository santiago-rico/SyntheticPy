import numpy as np
import pandas as pd


class SynthTables:
    def get_weights_table(self):
        control_units = self._control_units
        control_weights = self._control_weights

        weights_df = pd.DataFrame(
            dict(Unit=control_units, Weights=np.around(control_weights, 2))
        )
        return weights_df

    def get_predictor_comparison(self):
        unscaled_treated_predictors = self._unscaled_treated_predictors.ravel()
        unscaled_control_predictors = self._unscaled_control_predictors
        treated_unit = str(self._treated_unit)
        synth_treated_unit = "Synthetic " + str(treated_unit)

        comparison_df = pd.DataFrame(
            {
                treated_unit: unscaled_treated_predictors,
                synth_treated_unit: (
                    unscaled_control_predictors @ self._control_weights
                ).ravel(),
            },
            index=self._predictors,
        )
        return comparison_df
