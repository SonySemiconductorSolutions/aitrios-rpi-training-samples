from packaging import version

import tensorflow as tf

IS_FT_OR_OVER_2_13 = version.parse(tf.__version__) >= version.parse("2.13")
IS_FT_OR_UNDER_2_7 = version.parse(tf.__version__) <= version.parse("2.7")
