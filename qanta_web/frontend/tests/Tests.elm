module Tests exposing (..)

import Test exposing (..)
import Expect
import String
import Json.Decode exposing (decodeString)
import App exposing (Question, questionDecoder, buzzDecoder, gameStateDecoder, playerDecoder)


all : Test
all =
    describe "A Test Suite"
        [ test "Parse question" <|
            \() ->
                let
                    parsedQuestion =
                        decodeString questionDecoder
                            """
                            {
                                "id": 0,
                                "qb_id": 0,
                                "text": "This american president",
                                "answer": null,
                                "is_end_of_question": false
                            }
                            """
                in
                    case parsedQuestion of
                        Ok question ->
                            Expect.equal
                                question
                                { id = 0
                                , qb_id = 0
                                , text = "This american president"
                                , answer = Nothing
                                , isEndOfQuestion = False
                                }

                        Err error ->
                            Expect.fail error
        , test "Parse Buzz" <|
            \() ->
                let
                    parsedBuzz =
                        decodeString buzzDecoder
                            """
                            {
                                "player_id": 0,
                                "correct": false,
                                "guess": "abraham_lincoln"
                            }
                            """
                in
                    case parsedBuzz of
                        Ok buzz ->
                            Expect.equal
                                buzz
                                { playerId = 0
                                , correct = False
                                , guess = "abraham_lincoln"
                                }

                        Err error ->
                            Expect.fail error
        , test "Parse Player" <|
            \() ->
                let
                    parsedPlayer =
                        decodeString playerDecoder
                            """
                            {
                                "id": 0,
                                "name": "pedro",
                                "score": 10,
                                "answer_status": "unanswered",
                                "is_human": true
                            }
                            """
                in
                    case parsedPlayer of
                        Ok player ->
                            Expect.equal
                                player
                                { id = 0
                                , name = "pedro"
                                , score = 10
                                , answerStatus = App.Unanswered
                                , isHuman = True
                                }

                        Err error ->
                            Expect.fail error
        , test "Parse GameState" <|
            \() ->
                let
                    parsedGameState =
                        decodeString gameStateDecoder
                            """
                            {
                                "game_id": 0,
                                "players": [
                                    {"id": 1, "name": "pedro", "score": 0, "answer_status": "correct", "is_human": true},
                                    {"id": 2, "name": "jordan", "score": 100, "answer_status": "wrong", "is_human": true},
                                    {"id": 3, "name": "qanta", "score": 50, "answer_status": "unanswered", "is_human": false}
                                ],
                                "buzzes": [
                                    {"player_id": 1, "correct": false, "guess": ""},
                                    {"player_id": 3, "correct": false, "guess": "george_washington"}
                                ],
                                "question": {
                                    "id": 0,
                                    "qb_id": 0,
                                    "text": "This american president",
                                    "answer": null,
                                    "is_end_of_question": false
                                }
                            }
                            """
                in
                    case parsedGameState of
                        Ok gameState ->
                            Expect.equal
                                gameState
                                { gameId = 0
                                , players =
                                    [ { id = 1, name = "pedro", score = 0, answerStatus = App.Correct, isHuman = True }
                                    , { id = 2, name = "jordan", score = 100, answerStatus = App.Wrong, isHuman = True }
                                    , { id = 3, name = "qanta", score = 50, answerStatus = App.Unanswered, isHuman = False }
                                    ]
                                , buzzes =
                                    [ { playerId = 1, correct = False, guess = "" }
                                    , { playerId = 3, correct = False, guess = "george_washington" }
                                    ]
                                , question =
                                    Just
                                        { id = 0
                                        , qb_id = 0
                                        , text = "This american president"
                                        , answer = Nothing
                                        , isEndOfQuestion = False
                                        }
                                }

                        Err error ->
                            Expect.fail error
        ]
