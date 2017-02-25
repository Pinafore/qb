# Qanta Web Application

This is a web application that we use for exhibition matches playing humans against qanta. It is
also architected so that AI's can play against each other and that at some point in the future it
can be run as a public website where anyone can play against qanta.

## Architecture

The frontend UI code is written in [Elm](http://elm-lang.org) and the backend is written in python
using [Django](http://djangoproject.com). In general, we model this game interaction as a first
order Markov Chain. The UI is responsible for rendering the state into something pretty, and
collecting human actions. These are relayed to the backend game server which is responsible for
validating those actions, and them applying them to the game state. If the game has not ended then
it will query Qanta (or any other AI) for actions given the current state. It will finally return
this updated state to the UI so that the human(s) can provide their action