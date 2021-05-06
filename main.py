import numpy as np
import pandas as pd


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

        self._cols_to_drop = [outcome_variable,
                              id_variable, time_variable] + drop_columns
        self._predictors = [
            col for col in data if col not in self._cols_to_drop]
        self._num_predictors = len(self._predictors)
        self._num_pre_treatment_periods = data[data[time_variable]
                                               < time_of_treatment][time_variable].nunique()
        self._num_control_units = data[data[id_variable]
                                       != treated_unit][id_variable].nunique()

        self._treated_predictors = None
        self._control_predictors = None
        self._treated_outcome_before = None
        self._control_outcome_before = None
        self._treated_outcome_after = None
        self._control_outcome_after = None

    def _process_data(self):

        self._treated_predictors, self._treated_outcome_before, self._treated_outcome_after = self._get_treated_data()

        self._control_predictors, self._control_outcome_before, self._control_outcome_after = self._get_control_data()

    def _get_treated_data(self):
        treated = self._data[self._data[self._id_variable]
                             == self._treated_unit]

        # Treated predictors (For simplicity I just name them "treated_predictors" without specifying a period given that predictors don't play a role in our estimation after the treatment )
        treated_predictors = np.array(treated[treated[self._time_variable] < self._time_of_treatment][self._predictors].mean(
            axis=0)).reshape(self._num_predictors, 1)

        # Get the treated outcome before the time of the treatment
        treated_outcome_before = np.array(
            treated[treated[self._time_variable] < self._time_of_treatment][self._outcome_variable]).reshape(
            self._num_pre_treatment_periods, 1)

        # Get the treated outcome after the time of the treatment
        treated_outcome_after = np.array(
            treated[treated[self._time_variable] >= self._time_of_treatment][self._outcome_variable])

        return treated_predictors, treated_outcome_before, treated_outcome_after

    def _get_control_data(self):
        control = self._data[self._data[self._id_variable]
                             != self._treated_unit]

        # Create KxJ matrix
        before_predictors = control[control[self._time_variable]
                                    < self._time_of_treatment][self._predictors]

        control_predictors = np.array(
            before_predictors.set_index(np.arange(0, len(
                before_predictors)) // self._num_pre_treatment_periods).mean(level=0).transpose())

        # Get control outcomes
        control_outcome_before = np.array(
            control[control[self._time_variable] < self._time_of_treatment][
                self._outcome_variable]).reshape(self._num_control_units, self._num_pre_treatment_periods).transpose()

        control_outcome_after = np.array(
            control[control[self._time_variable] >= self._time_of_treatment][
                self._outcome_variable
            ]
        )

        return control_predictors, control_outcome_before, control_outcome_after,


class SyntheticControl(DataPrep):
    def __init__(
        self,
        data,
        outcome_variable,
        id_variable,
        time_variable,
        time_of_treatment,
        treated_unit,
        drop_columns=[],
    ):
        super().__init__(data, outcome_variable, id_variable,
                         time_variable, time_of_treatment, treated_unit, drop_columns)
        self._process_data()


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
    )

    # print(synth._treated_predictors)
    # print(synth.treated_outcome_before)
    # print(synth.control_predictors)
    # print(synth.control_outcome_before)
