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
        self.data = data
        self.outcome_variable = outcome_variable
        self.id_variable = id_variable
        self.time_variable = time_variable
        self.time_of_treatment = time_of_treatment
        self.treated_unit = treated_unit
        self.drop_columns = drop_columns

        self.cols_to_drop = [outcome_variable,
                             id_variable, time_variable] + drop_columns
        self.predictors = [col for col in data if col not in self.cols_to_drop]
        self.num_predictors = len(self.predictors)
        self.num_pre_treatment_periods = data[data[time_variable]
                                              < time_of_treatment][time_variable].nunique()
        self.num_control_units = data[data[id_variable]
                                      != treated_unit][id_variable].nunique()

        self.treated_predictors = None
        self.control_predictors = None
        self.treated_outcome_before = None
        self.control_outcome_before = None
        self.treated_outcome_after = None
        self.control_outcome_after = None

    def _process_data(self):

        self.treated_predictors, self.treated_outcome_before, self.treated_outcome_after = self._get_treated_data()

        self.control_predictors, self.control_outcome_before, self.control_outcome_after = self._get_control_data()

    def _get_treated_data(self):
        treated = self.data[self.data[self.id_variable] == self.treated_unit]

        # Treated predictors (For simplicity I just name them "treated_predictors" without specifying a period given that predictors don't play a role in our estimation after the treatment )
        treated_predictors = np.array(treated[treated[self.time_variable] < self.time_of_treatment][self.predictors].mean(
            axis=0)).reshape(self.num_predictors, 1)

        # Get the treated outcome before the time of the treatment
        treated_outcome_before = np.array(
            treated[treated[self.time_variable] < self.time_of_treatment][self.outcome_variable]).reshape(
            self.num_pre_treatment_periods, 1)

        # Get the treated outcome after the time of the treatment
        treated_outcome_after = np.array(
            treated[treated[self.time_variable] >= self.time_of_treatment][self.outcome_variable])

        return treated_predictors, treated_outcome_before, treated_outcome_after

    def _get_control_data(self):
        control = self.data[self.data[self.id_variable] != self.treated_unit]

        # Create KxJ matrix
        before_predictors = control[control[self.time_variable]
                                    < self.time_of_treatment][self.predictors]

        control_predictors = np.array(
            before_predictors.set_index(np.arange(0, len(
                before_predictors)) // self.num_pre_treatment_periods).mean(level=0).transpose())

        # Get control outcomes
        control_outcome_before = np.array(
            control[control[self.time_variable] < self.time_of_treatment][
                self.outcome_variable]).reshape(self.num_control_units, self.num_pre_treatment_periods).transpose()

        control_outcome_after = np.array(
            control[control[self.time_variable] >= self.time_of_treatment][
                self.outcome_variable
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

    # print(synth.treated_predictors)
    # print(synth.treated_outcome_before)
    # print(synth.control_predictors)
    # print(synth.control_outcome_before)
