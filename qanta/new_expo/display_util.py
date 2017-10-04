import json
import textwrap
from collections import defaultdict, Counter, namedtuple
import argparse
import itertools
from csv import DictReader
from time import sleep
import os

kBIGNUMBERS = {-1:
"""








88888888
88888888





""",
0:
"""

    .n~~%x.
  x88X   888.
 X888X   8888L
X8888X   88888
88888X   88888X
88888X   88888X
88888X   88888f
48888X   88888
 ?888X   8888"
  "88X   88*`
    ^"==="`



""",
1:
"""

      oe
    .@88
==*88888
   88888
   88888
   88888
   88888
   88888
   88888
   88888
'**%%%%%%**



""",
2:
"""

  .--~*teu.
 dF     988Nx
d888b   `8888>
?8888>  98888F
 "**"  x88888~
      d8888*`
    z8**"`   :
  :?.....  ..F
 <""888888888~
 8:  "888888*
 ""    "**"`



""",
3:
"""

  .x~~"*Weu.
 d8Nu.  9888c
 88888  98888
 "***"  9888%
      ..@8*"
   ````"8Weu
  ..    ?8888L
:@88N   '8888N
*8888~  '8888F
'*8"`   9888%
  `~===*%"`



""",
4:
"""

        xeee
       d888R
      d8888R
     @ 8888R
   .P  8888R
  :F   8888R
 x"    8888R
d8eeeee88888eer
       8888R
       8888R
    "*%%%%%%**~



""",
5:
"""

  cuuu....uK
  888888888
  8*888**"
  >  .....
  Lz"  ^888Nu
  F     '8888k
  ..     88888>
 @888L   88888
'8888F   8888F
 %8F"   d888"
  ^"===*%"`



""",
6:
"""

    .ue~~%u.
  .d88   z88i
 x888E  *8888
:8888E   ^""
98888E.=tWc.
98888N  '888N
98888E   8888E
'8888E   8888E
 ?888E   8888"
  "88&   888"
    ""==*""



""",
7:
"""

dL ud8Nu  :8c
8Fd888888L %8
4N88888888cuR
4F   ^""%""d
d       .z8
^     z888
    d8888'
   888888
  :888888
   888888
   '%**%



""",
8:
"""

   u+=~~~+u.
 z8F      `8N.
d88L       98E
98888bu.. .@*
"88888888NNu.
 "*8888888888i
 .zf""*8888888L
d8F      ^%888E
88>        `88~
'%N.       d*"
   ^"====="`



""",
9:
"""

  .xn!~%x.
 x888   888.
X8888   8888:
88888   X8888
88888   88888>
`8888  :88888X
  `"**~ 88888>
 .xx.   88888
'8888>  8888~
 888"  :88%
  ^"===""


"""}

class kCOLORS:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print(text, color="RED", end='\n'):
        start = getattr(kCOLORS, color)
        print(start + text + kCOLORS.ENDC, end=end)


def clear_screen():
    print("Clearing")
    os.system('cls' if os.name == 'nt' else 'clear')

def show_score(left_score, right_score,
               left_header="HUMAN", right_header="COMPUTER",
               left_color="GREEN", right_color="BLUE",
               flush=True):
    assert isinstance(left_score, int)
    assert isinstance(right_score, int)
    if flush:
        clear_screen()
    # Print the header
    print("%-15s" % "", end='')
    kCOLORS.print("%-15s" % left_header, left_color, end='')
    print("%-30s" % "", end='')
    kCOLORS.print("%-15s\n" % right_header, right_color)

    for line in range(1, 15):
        for num, color in [(left_score, left_color),
                           (right_score, right_color)]:
            for place in [100, 10, 1]:
                if place == 100 and num < 0:
                    val = -1
                else:
                    val = (abs(num) % (place * 10)) // place
                kCOLORS.print("%-15s" % kBIGNUMBERS[val].split("\n")[line],
                              color=color, end=' ')
            print("|", end=" ")
        print(" ")
