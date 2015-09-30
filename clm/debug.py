from lm_wrapper import LanguageModelReader

if __name__ == "__main__":
    lm = LanguageModelReader("lm.txt")
    lm.init()

    chicago = """ This city houses a {Zaha Hadid} [zah-hah hah-DEED]-designed {pavilion}
that was named for a man who developed a 1909 plan for it, {Daniel Burnham}.
A music venue sits under a curving metal framework at its Pritzker Pavilion,
while {cantilever}ed {roof}s grace its {Robie House}.  The {John Hancock
Center} is in this city, as is the {tallest building} in the {United States}.
For 10 points--name this city of broad shoulders home of the Willis Tower, formerly called the Sears
Tower.  """

    earnest = """ Its two male leads eat Lane's cucumber sandwiches in the first scene, and
they fight over who will eat the muffins at the end of Act II.  One of them
was adopted by Thomas Cardew, whose niece became his ward.  The other relates
his (*) invented sick friend Bunbury to the former's invention of the title
wicked brother, whom they impersonate to impress Cecily and Gwendolen.  For 10
points--name this play by Oscar Wilde.  """

    good = u"""The family sees Stone Mountain and has barbequed sandwiches at The Tower, run by Red Sammy, then heads for Florida over the grandmother's objections.When Pitty Sing jumps on Bailey's shoulder, a car accident leaves them in the hands of Hiram, Bobby Lee, and an escaped (*) serial killer called The Misfit.The scarcity of moral males is asserted by the title of--for 10 points--what short story by Flannery O'Connor"""


    for guess, question in [
            ("The Importance of Being Earnest", earnest),
            ("Chicago", chicago), 
            ('A Good Man Is Hard to Find (short story)', good)]:
        for corpus in ["qb", "wiki", "source"]:
            print(lm.feature_line(corpus, guess, question))
