module Elements exposing (..)

import Html exposing (Html, text, div, img, li, ul, nav, button, a, Attribute, h3, span, p)
import Html.Attributes exposing (src, class, type_)


card : List (Html a) -> Html a
card content =
    div [ class "card" ] content


row : List (Html a) -> Html a
row content =
    div [ class "row" ] content


colmd : Int -> List (Html a) -> Html a
colmd colWidth content =
    div [ class ("col-md-" ++ (toString colWidth)) ] content


colmdoffset : Int -> Int -> List (Html a) -> Html a
colmdoffset colWidth offset content =
    div [ class ("col-md-" ++ (toString colWidth) ++ " col-md-offset-" ++ (toString offset)) ] content


jumbotron : String -> Maybe String -> String -> Html a
jumbotron classes header body =
    let
        headerHtml =
            case header of
                Just headerStr ->
                    [ h3 [] [ text headerStr ] ]

                Nothing ->
                    []
    in
        div [ class ("jumbotron " ++ classes) ]
            (headerHtml
                ++ [ p [] [ text body ] ]
            )


template : Html a -> Html a
template content =
    div [ class "wrapper" ]
        [ div [ class "main" ]
            [ container [ class "main-container" ] content ]
        ]


container : List (Attribute a) -> Html a -> Html a
container classes content =
    div ([ class "container" ] ++ classes) [ content ]


containerList : List (Attribute a) -> List (Html a) -> Html a
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
