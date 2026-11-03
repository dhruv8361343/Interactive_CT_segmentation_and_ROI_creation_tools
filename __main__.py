"""
Allow launching the desktop toolkit with `python -m ctseg`.

FILE PURPOSE:
    When a user runs `python -m ctseg`, Python looks for a `__main__.py`
    file inside the ctseg package. This file simply delegates to `app.main()`,
    which handles CLI argument parsing and either inspects a TIFF or launches
    the PyQt6 desktop application.

    `raise SystemExit(main())` is used instead of `sys.exit()` so that the
    exit code from main() (0 = success, nonzero = error) propagates correctly
    to the calling shell.
"""

from ctseg.app import main

# SystemExit propagates the integer return code from main() to the OS.
raise SystemExit(main())
