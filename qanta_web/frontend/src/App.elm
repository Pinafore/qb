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


type alias GameState =
    { gameId : Int
    , players : List Player
    , text : String
    , buzzes : List Buzz
    , answer : Maybe String
    , isEndOfQuestion : Bool
    }


init : String -> ( Model, Cmd Msg )
init path =
    ( { gameState = defaultGameState, keyPress = Nothing }, Cmd.none )


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
    let
        parsedState =
            case dummyState of
                Err msg ->
                    defaultGameState

                Ok state ->
                    state
    in
        div []
            [ navbar
            , (template
                (div []
                    [ row [ h3 [] [ text "Question 1" ] ]
                    , row [ (jumbotron "question-text" Nothing "This is the start of a question!") ]
                    , row [ renderBuzzStatus model ]
                    , row [ h3 [] [ text "Scoreboard" ] ]
                    , row (List.map renderPlayer parsedState.players)
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


renderAnswerStatus : AnswerStatus -> Html Msg
renderAnswerStatus status =
    case status of
        Unanswered ->
            span [ class "label label-info" ] [ text "Ready to Buzz" ]

        Correct ->
            span [ class "label label-success" ] [ text "Correct" ]

        Wrong ->
            span [ class "label label-danger" ] [ text "Wrong" ]


playerDecoder : Decoder Player
playerDecoder =
    decode Player
        |> required "id" int
        |> required "name" string
        |> required "score" int
        |> required "answer_status" answerStatusDecoder
        |> required "is_human" bool


defaultPlayer =
    { id = -1
    , name = "default"
    , score = 0
    , answered = False
    , isHuman = False
    }


buzzDecoder : Decoder Buzz
buzzDecoder =
    decode Buzz
        |> required "player_id" int
        |> required "correct" bool
        |> required "guess" string


defaultGameState =
    { gameId = 1
    , players = []
    , text = ""
    , buzzes = []
    , answer = Nothing
    , isEndOfQuestion = False
    }


gameStateDecoder : Decoder GameState
gameStateDecoder =
    decode GameState
        |> required "game_id" int
        |> required "players" (list playerDecoder)
        |> required "text" string
        |> required "buzzes" (list buzzDecoder)
        |> required "answer" (nullable string)
        |> required "is_end_of_question" bool


dummyState =
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
      "text": "Who was this american president who was born in a logged cabin?",
      "buzzes": [
        {"player_id": 1, "correct": false, "guess": ""},
        {"player_id": 3, "correct": false, "guess": "george_washington"}
      ],
      "answer": null,
      "is_end_of_question": false
    }
  """


dummyPlayer =
    Json.Decode.decodeString
        playerDecoder
        """
        {"id": 1, "name": "pedro", "score": 0, "answer_status": "correct", "is_human": true}
        """


playerList =
    let
        parsedState =
            case dummyState of
                Err msg ->
                    defaultGameState

                Ok state ->
                    state
    in
        List.map renderPlayer parsedState.players


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
