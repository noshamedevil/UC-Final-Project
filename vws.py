from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

# Used for ensembles using only deaths
class WindowSize(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, window_size, random_state=None):
        self.fitted_ = False
        self.window_size = window_size

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
            return X[:, -(self.window_size + 1):]

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'window_size',
                'name': 'window_size',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        window_size = UniformIntegerHyperparameter(
            name="window_size", lower=3, upper=30, default_value=14, log=False
        )
        cs.add_hyperparameters([window_size])
        return cs

# Used for ensembles using combined mobility
class WindowSizeMultiple(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, window_size, random_state=None):
        self.fitted_ = False
        self.window_size = window_size

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
            return X[:,-(self.window_size * 9 + 1):] # 9 == number of columns

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'window_size',
                'name': 'window_size',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        window_size = UniformIntegerHyperparameter(
            name="window_size", lower=3, upper=30, default_value=14, log=False
        )
        cs.add_hyperparameters([window_size])
        return cs

# Used for ensembles using mode of transportation mobility
class WindowSizeApple(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, window_size, random_state=None):
        self.fitted_ = False
        self.window_size = window_size

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
            return X[:,-(self.window_size * 3 + 1):] # 3 == number of columns

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'window_size',
                'name': 'window_size',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        window_size = UniformIntegerHyperparameter(
            name="window_size", lower=3, upper=30, default_value=14, log=False
        )
        cs.add_hyperparameters([window_size])
        return cs

# Used for ensembles using place visits mobility
class WindowSizeGoogle(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, window_size, random_state=None):
        self.fitted_ = False
        self.window_size = window_size

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
            return X[:,-(self.window_size * 7 + 1):] # 7 == number of columns

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'window_size',
                'name': 'window_size',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        window_size = UniformIntegerHyperparameter(
            name="window_size", lower=3, upper=30, default_value=14, log=False
        )
        cs.add_hyperparameters([window_size])
        return cs


class WindowSizeWavelet(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, window_size, random_state=None):
        self.fitted_ = False
        self.window_size = window_size

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
            return X[:,-(self.window_size + 1):]

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'window_size',
                'name': 'window_size',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        window_size = UniformIntegerHyperparameter(
            name="window_size", lower=1, upper=15, default_value=7, log=False
        )
        cs.add_hyperparameters([window_size])
        return cs