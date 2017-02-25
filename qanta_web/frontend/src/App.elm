module App exposing (..)

import Html exposing (Html, text, div, img, li, ul, nav, button, a, Attribute, h3)
import Html.Attributes exposing (src, class, type_)
import Json.Decode exposing (int, string, float, bool, list, nullable, Decoder)
import Json.Decode.Pipeline exposing (decode, required, optional)


type alias Model =
    { message : String
    , logo : String
    }


type alias Player =
    { id : Int
    , name : String
    , score : Int
    , answered : Bool
    , isHuman : Bool
    }


playerDecoder : Decoder Player
playerDecoder =
    decode Player
        |> required "id" int
        |> required "name" string
        |> required "score" int
        |> required "answered" bool
        |> required "is_human" bool


defaultPlayer =
    { id = -1
    , name = "default"
    , score = 0
    , answered = False
    , isHuman = False
    }


type alias Buzz =
    { playerId : Int
    , correct : Bool
    , guess : String
    }


buzzDecoder : Decoder Buzz
buzzDecoder =
    decode Buzz
        |> required "player_id" int
        |> required "correct" bool
        |> required "guess" string


type alias GameState =
    { gameId : Int
    , players : List Player
    , text : String
    , buzzes : List Buzz
    , answer : Maybe String
    , isEndOfQuestion : Bool
    }


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
        {"id": 1, "name": "pedro", "score": 0, "answered": true, "is_human": true},
        {"id": 2, "name": "jordan", "score": 100, "answered": false, "is_human": true},
        {"id": 3, "name": "qanta", "score": 50, "answered": false, "is_human": false}
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
        {"id": 1, "name": "pedro", "score": 0, "answered": false, "is_human": true}
        """


init : String -> ( Model, Cmd Msg )
init path =
    ( { message = "Your Elm App is working!", logo = path }, Cmd.none )


type Msg
    = NoOp


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
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
                    [ row (List.map renderPlayer parsedState.players)
                    ]
                )
              )
            ]


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
                ]
            ]
        ]


template : Html Msg -> Html Msg
template content =
    div [ class "wrapper" ]
        [ div [ class "main" ]
            [ container [ class "main-container" ] content ]
        ]


container : List (Attribute Msg) -> Html Msg -> Html Msg
container classes content =
    div ([ class "container" ] ++ classes) [ content ]


containerList : List (Attribute Msg) -> List (Html Msg) -> Html Msg
containerList classes contentList =
    div ([ class "container" ] ++ classes) contentList


navbar =
    nav [ class "navbar navbar-info navbar-fixed-top" ]
        [ containerList []
            [ div [ class "navbar-header" ] []
            , div [ class "collapse navbar-collapse" ]
                [ ul [ class "nav navbar-nav text-center" ]
                    [ li [] [ h3 [] [ text "QANTA AI Exhibition" ] ]
                    ]
                ]
            ]
        ]


card : List (Html Msg) -> Html Msg
card content =
    div [ class "card" ] content


row : List (Html Msg) -> Html Msg
row content =
    div [ class "row" ] content


colmd : Int -> List (Html Msg) -> Html Msg
colmd colWidth content =
    div [ class ("col-md-" ++ (toString colWidth)) ] content


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.none
