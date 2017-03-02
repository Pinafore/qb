module App exposing (..)

import Html exposing (Html, text, div, img, li, ul, nav, button, a, Attribute, h3, span, p, h4)
import Html.Attributes exposing (src, class, type_)
import Keyboard exposing (downs)
import Json.Decode as JsonDecode
import Json.Decode exposing (int, string, float, bool, list, nullable, Decoder, fail, succeed, andThen)
import Json.Decode.Pipeline exposing (decode, required, optional)
import Char exposing (fromCode)
import Elements exposing (colmd, colmdoffset, card, navbar, jumbotron, row, template)


type Msg
    = NoOp
    | HumanBuzz String
    | Presses Char


type alias Model =
    { gameState : GameState
    , keyPress : Maybe Char
    }


type alias Player =
    { id : Int
    , name : String
    , score : Int
    , answerStatus : AnswerStatus
    , isHuman : Bool
    }


type AnswerStatus
    = Unanswered
    | Correct
    | Wrong


type alias Buzz =
    { playerId : Int
    , correct : Bool
    , guess : String
    }


type alias Question =
    { id : Int
    , qb_id : Int
    , text : String
    , answer : Maybe String
    , isEndOfQuestion : Bool
    }


type alias GameState =
    { gameId : Int
    , players : List Player
    , buzzes : List Buzz
    , question : Maybe Question
    }


init : String -> ( Model, Cmd Msg )
init path =
    ( { gameState = dummyState, keyPress = Nothing }, Cmd.none )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        HumanBuzz player ->
            ( model, Cmd.none )

        Presses char ->
            ( model, Cmd.none )


view : Model -> Html Msg
view model =
    div []
        [ navbar
        , (template
            (div []
                [ row [ h3 [] [ text "Question 1" ] ]
                , row [ (jumbotron "question-text" Nothing "This is the start of a question!") ]
                , row [ renderBuzzStatus model ]
                , row [ h3 [] [ text "Scoreboard" ] ]
                , row (List.map renderPlayer dummyState.players)
                ]
            )
          )
        ]


subscriptions : Model -> Sub Msg
subscriptions model =
    Keyboard.downs (\code -> Presses (fromCode code))


type AlertType
    = Info
    | Success
    | Warning
    | Danger


alert : AlertType -> String -> Html a -> Html a
alert alertType classes content =
    let
        alertCss =
            case alertType of
                Info ->
                    "alert-info"

                Success ->
                    "alert-success"

                Warning ->
                    "alert-warning"

                Danger ->
                    "alert-danger"
    in
        div [ class ("alert " ++ alertCss ++ " " ++ classes) ] [ div [ class "container-fluid" ] [ content ] ]


renderBuzzStatus : Model -> Html Msg
renderBuzzStatus model =
    colmdoffset 6 3 [ alert Success "buzz-status" (h4 [] [ text "Status: No Buzzes Yet!" ]) ]


answerStatusDecoder : Decoder AnswerStatus
answerStatusDecoder =
    let
        answerStatusParse : String -> Decoder AnswerStatus
        answerStatusParse raw =
            case raw of
                "unanswered" ->
                    succeed Unanswered

                "correct" ->
                    succeed Correct

                "wrong" ->
                    succeed Wrong

                _ ->
                    fail "could not parse answer status"
    in
        string |> andThen answerStatusParse


playerDecoder : Decoder Player
playerDecoder =
    decode Player
        |> required "id" int
        |> required "name" string
        |> required "score" int
        |> required "answer_status" answerStatusDecoder
        |> required "is_human" bool


buzzDecoder : Decoder Buzz
buzzDecoder =
    decode Buzz
        |> required "player_id" int
        |> required "correct" bool
        |> required "guess" string


gameStateDecoder : Decoder GameState
gameStateDecoder =
    decode GameState
        |> required "game_id" int
        |> required "players" (list playerDecoder)
        |> required "buzzes" (list buzzDecoder)
        |> required "question" (nullable questionDecoder)


questionDecoder : Decoder Question
questionDecoder =
    decode Question
        |> required "id" int
        |> required "qb_id" int
        |> required "text" string
        |> required "answer" (nullable string)
        |> required "is_end_of_question" bool


renderAnswerStatus : AnswerStatus -> Html Msg
renderAnswerStatus status =
    case status of
        Unanswered ->
            span [ class "label label-info" ] [ text "Ready to Buzz" ]

        Correct ->
            span [ class "label label-success" ] [ text "Correct" ]

        Wrong ->
            span [ class "label label-danger" ] [ text "Wrong" ]


defaultPlayer : Player
defaultPlayer =
    { id = -1
    , name = "default"
    , score = 0
    , answerStatus = Unanswered
    , isHuman = False
    }


defaultGameState : GameState
defaultGameState =
    { gameId = 1
    , players = []
    , buzzes = []
    , question = Just { id = 0, qb_id = 0, text = "", answer = Just "albert_einstein", isEndOfQuestion = False }
    }


dummyState : GameState
dummyState =
    let
        parsedGameState =
            Json.Decode.decodeString
                gameStateDecoder
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
                gameState

            Err _ ->
                defaultGameState


dummyPlayer : Player
dummyPlayer =
    let
        parsedPlayer =
            Json.Decode.decodeString
                playerDecoder
                """
            {"id": 1, "name": "pedro", "score": 0, "answer_status": "correct", "is_human": true}
            """
    in
        case parsedPlayer of
            Ok player ->
                player

            Err _ ->
                defaultPlayer


renderPlayerList : List (Html Msg)
renderPlayerList =
    List.map renderPlayer dummyState.players


renderPlayer : Player -> Html Msg
renderPlayer player =
    colmd 3
        [ card
            [ div [ class "player-card" ]
                [ div [] [ text ("Player: " ++ player.name) ]
                , div [] [ text ("Score: " ++ toString player.score) ]
                , div [] [ renderAnswerStatus player.answerStatus ]
                ]
            ]
        ]
