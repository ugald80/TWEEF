import logging
import warnings
from TWEEF.Logging import TWEEFLogger
from TWEEF.Aggregation import TRUSTWORTHINESS_SCORE_SCALE
from TWEEF.TrustworthyClassifier import TrustworthyClassifier
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress AIF360 warnings
logging.getLogger("aif360").setLevel(logging.ERROR)

# Ignore scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "TRUSTWORTHINESS_SCORE_SCALE",
    "TrustworthyClassifier",
    "TWEEFLogger",
]
