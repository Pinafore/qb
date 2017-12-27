import dash
import dash_core_components as dcc
import dash_html_components as html


def main():
    app = dash.Dash()
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='Dash: a web app framework'),

        dcc.Graph(
            id='example',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'}
                ],
                'layout': {
                    'title': 'Dash Visualization'
                }
            }
        )
    ])
    app.run_server(debug=True)


if __name__ == '__main__':
    main()