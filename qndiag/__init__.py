# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: MIT
"""Joint diagonalization in Python"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

__version__ = '0.1'

from .qndiag import qndiag, transform_set, loss, gradient  # noqa
from .pham import ajd_pham  # noqa
