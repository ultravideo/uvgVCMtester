"""Entry point.

Works both as `python -m vcmtest ...` and as `python vcmtest ...` (running the
package directory as a script, as in the README). In the latter case the
package's parent directory is not on sys.path, so add it before importing.
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcmtest.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
