import os
from imx500_zoo.utilities.third_party import setup_path

setup_path()

from imx500_zoo.imx500_zoo import (
    Solution,
    main_imx500_zoo,
    main_cli,
)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
