class GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            sys.stdin.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

getch = GetchUnix()

def interpret_keypress():
    """
    See whether a number was pressed (give terminal bell if so) and return
    value.  Otherwise returns none.  Tries to handle arrows as a single
    press.
    """
    press = getch()

    if press == 'Q':
        raise Exception('Exiting expo by user request from pressing Q')

    if press == '\x1b':
        getch()
        getch()
        press = "direction"

    if press != "direction" and press != " ":
        try:
            press = int(press)
        except ValueError:
            press = None
    return press

