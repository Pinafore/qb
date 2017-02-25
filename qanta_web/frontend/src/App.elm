module App exposing (..)

import Html exposing (Html, text, div, img, li, ul, nav, button, a)
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
    div []
        [ navbar
        , (template
            (div []
                [ img [ src model.logo ] []
                , let
                    playerHtml =
                        case dummyPlayer of
                            Err msg ->
                                text "An error occurred"

                            Ok player ->
                                card [ renderPlayer player ]
                  in
                    row [ playerHtml ]
                ]
            )
          )
        ]


renderPlayer : Player -> Html Msg
renderPlayer player =
    div []
        [ text (player |> .id |> toString)
        , text player.name
        ]


template : Html Msg -> Html Msg
template content =
    div [ class "wrapper" ]
        [ div [ class "main" ]
            [ container content ]
        ]


container : Html Msg -> Html Msg
container content =
    div [ class "container" ] [ content ]


containerList : List (Html Msg) -> Html Msg
containerList contentList =
    div [ class "container" ] contentList


navbar =
    nav [ class "navbar navbar-info navbar-fixed-top" ]
        [ containerList
            [ div [ class "navbar-header" ] []
            , div [ class "collapse navbar-collapse" ]
                [ ul [ class "nav navbar-nav navbar-right" ]
                    [ li [] [ a [] [ text "Navigation" ] ]
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
