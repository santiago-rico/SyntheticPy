import numpy as np
import pandas as pd
from solver import Solver
from tables import SynthTables


class DataPrep:
    def __init__(
        self,
        data,
        outcome_variable,
        id_variable,
        time_variable,
        time_of_treatment,
        treated_unit,
        drop_columns,
    ):
        self._data = data
        self._outcome_variable = outcome_variable
        self._id_variable = id_variable
        self._time_variable = time_variable
        self._time_of_treatment = time_of_treatment
        self._treated_unit = treated_unit

        # Basic checks
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError("data argument is not a pandas DataFrame")

        if not isinstance(self._outcome_variable, str):
            raise TypeError("outcome_variable argument is not a string")

        if self._outcome_variable not in self._data.columns:
            raise KeyError(
                f"{outcome_variable} is not a column of the dataset"
            )

        if not isinstance(self._id_variable, str):
            raise TypeError("id_variable argument is not a string")

        if self._id_variable not in self._data.columns:
            raise KeyError(f"{id_variable} is not a column of the dataset")

        if self._treated_unit not in self._data[self._id_variable].unique():
            raise Exception(
                f"treated_unit argument not found in the {id_variable} column"
            )

        non_found_drop = [
            col for col in drop_columns if col not in self._data.columns
        ]
        if len(non_found_drop) > 0:
            raise Exception(
                f"{','.join(non_found_drop)} not found in dataset columns"
            )

        # SyntheticControl(
        # [1, 2, 3],
        # # germany,
        # "gdp",
        # "country",
        # "year",
        # 1990,
        # "West Germany",
        # drop_columns=["index"]

        self._cols_to_drop = [
            outcome_variable,
            id_variable,
            time_variable,
        ] + drop_columns
        self._predictors = [
            col for col in data if col not in self._cols_to_drop
        ]
        self._num_predictors = len(self._predictors)
        self._num_pre_treatment_periods = data[
            data[time_variable] < time_of_treatment
        ][time_variable].nunique()
        self._num_after_treatment_periods = data[
            data[time_variable] >= time_of_treatment
        ][time_variable].nunique()
        self._num_control_units = data[data[id_variable] != treated_unit][
            id_variable
        ].nunique()
        self._control_units = data[data[id_variable] != treated_unit][
            id_variable
        ].unique()

        self._treated_predictors = None
        self._control_predictors = None
        self._treated_outcome_before = None
        self._control_outcome_before = None
        self._treated_outcome_after = None
        self._control_outcome_after = None

    def _process_data(self):

        (
            self._unscaled_treated_predictors,
            self._treated_outcome_before,
            self._treated_outcome_after,
        ) = self._get_treated_data()

        (
            self._unscaled_control_predictors,
            self._control_outcome_before,
            self._control_outcome_after,
        ) = self._get_control_data()

        (
            self._treated_predictors,
            self._control_predictors,
        ) = self._rescale_predictors()

    def _get_treated_data(self):
        treated = self._data[
            self._data[self._id_variable] == self._treated_unit
        ]

        # For simplicity I just name them "treated_predictors" without
        # specifying a period given that predictors don't play a role
        # in our estimation after the # treatment)
        treated_predictors = np.array(
            treated[treated[self._time_variable] < self._time_of_treatment][
                self._predictors
            ].mean(axis=0)
        ).reshape(self._num_predictors, 1)

        # Get the treated outcome before the time of the treatment
        treated_outcome_before = np.array(
            treated[treated[self._time_variable] < self._time_of_treatment][
                self._outcome_variable
            ]
        ).reshape(self._num_pre_treatment_periods, 1)

        # Get the treated outcome after the time of the treatment
        treated_outcome_after = np.array(
            treated[treated[self._time_variable] >= self._time_of_treatment][
                self._outcome_variable
            ]
        ).reshape(self._num_after_treatment_periods, 1)

        return (
            treated_predictors,
            treated_outcome_before,
            treated_outcome_after,
        )

    def _get_control_data(self):
        control = self._data[
            self._data[self._id_variable] != self._treated_unit
        ]

        # Create KxJ matrix
        before_predictors = control[
            control[self._time_variable] < self._time_of_treatment
        ][self._predictors]

        control_predictors = np.array(
            before_predictors.set_index(
                np.arange(0, len(before_predictors))
                // self._num_pre_treatment_periods
            )
            .mean(level=0)
            .transpose()
        )

        # Get control outcomes
        control_outcome_before = (
            np.array(
                control[
                    control[self._time_variable] < self._time_of_treatment
                ][self._outcome_variable]
            )
            .reshape(self._num_control_units, self._num_pre_treatment_periods)
            .transpose()
        )

        control_outcome_after = (
            np.array(
                control[
                    control[self._time_variable] >= self._time_of_treatment
                ][self._outcome_variable]
            )
            .reshape(
                self._num_control_units, self._num_after_treatment_periods
            )
            .transpose()
        )

        return (
            control_predictors,
            control_outcome_before,
            control_outcome_after,
        )

    def _rescale_predictors(self):
        all_predictors = np.concatenate(
            (
                self._unscaled_treated_predictors,
                self._unscaled_control_predictors,
            ),
            axis=1,
        )
        all_predictors /= np.apply_along_axis(np.std, 0, all_predictors)

        # we know treated is in the first column since we concatenated it
        treated_preds = all_predictors[:, 0]
        control_preds = all_predictors[:, 1:]  # all other columns are control

        return treated_preds, control_preds


class SyntheticControl(DataPrep, Solver, SynthTables):
    def __init__(
        self,
        data,
        outcome_variable,
        id_variable,
        time_variable,
        time_of_treatment,
        treated_unit,
        drop_columns=[],
        custom_v=[],
    ):
        super().__init__(
            data,
            outcome_variable,
            id_variable,
            time_variable,
            time_of_treatment,
            treated_unit,
            drop_columns,
        )

        self._process_data()
        self._custom_v = custom_v

        (
            self.treated_outcome_estimate,
            self._predictors_importance,
            self._control_weights,
        ) = self.estimate(
            self._treated_predictors,
            self._control_predictors,
            self._treated_outcome_before,
            self._control_outcome_before,
            self._control_outcome_after,
            self._custom_v,
        )


if __name__ == "__main__":
    germany = pd.read_stata("repgermany.dta")
    synth = SyntheticControl(
        germany,
        "gdp",
        "country",
        "year",
        1990,
        "West Germany",
        drop_columns=["index"],
        # custom_v=[
        #     0.02330873,
        #     0.3640076,
        #     0.01755632,
        #     0.0641354,
        #     0.3532059,
        #     0.1709646,
        #     0.006821504,
        # ],
    )

    print(synth.get_weights_table())
    print(synth.get_predictor_comparison())
    print(synth._predictors_importance)
    # print(synth._treated_outcome_before)
    # print(synth._control_predictors)
    # print(synth._control_outcome_before)
