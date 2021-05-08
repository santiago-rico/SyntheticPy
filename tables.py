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
