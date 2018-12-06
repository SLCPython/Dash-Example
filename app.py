# Example Dash app using Scikit-Learn Iris Dataset
# SLC Python Meetup
# 2018-12-05

# %% Install/Import packages

# pip install dash==0.31.1
# pip install dash-html-components==0.13.2
# pip install dash-core-components==0.39.0
# pip install plotly
# pip install pandas
# pip install scikit-learn

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from sklearn.datasets import load_iris

# %% Import and Clean Data

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.concat([df, pd.Series(data.target, name='iris_class')], 1)
df = df.iloc[:, 2:]
class_name_map = {0: 'Setosa',
                  1: 'Versicolor',
                  2: 'Virginica'}
df = df.assign(iris_class=df.iris_class.apply(lambda x: class_name_map[x]))

class_counts = df.iris_class.value_counts()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# %% Define App

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Build Layout
app.layout = html.Div(children=[
    # HTML tags
    html.H1(['Exploring Iris Dataset']),
    # Dash Core Components
    dcc.Markdown(["""This is the Iris Dataset. There are 3 classes of Iris:
    *Setosa*, *Versicolor*, and *Virginica*.  
    """]),
    html.Br(),
    dcc.Graph(id='graph1',
              figure={'data': [go.Bar(x=class_counts.index,
                                      y=class_counts.values,
                                      text=class_counts.values,
                                      textposition='auto')],
                      'layout': {'title': 'Overall Iris Class Counts'}}),
    html.Br(),
    dcc.Graph(id='graph2'),
    html.Br(),
    dcc.Dropdown(
        id='dropdown-state',
        options=[
            {'label': 'Setosa', 'value': 'Setosa'},
            {'label': 'Virginica', 'value': 'Virginica'},
            {'label': 'Versicolor', 'value': 'Versicolor'}
        ],
        value='Setosa'
    ),
    html.Button(id='submit-button', n_clicks=0, children='submit'),
    html.Br(),
    html.Table(id='table1')
])


# Build a Callback
@app.callback(Output(component_id='graph2', component_property='figure'),
              [Input('graph1', 'clickData')])
def print_click_data(clickdata):
    iris_class = (clickdata['points'][0]['x']
                  if clickdata is not None else 'Setosa')
    dff = df[df.iris_class == iris_class]
    chart = {
        'data': [go.Scatter(x=dff.iloc[:, 0],
                            y=dff.iloc[:, 1],
                            mode='markers',
                            marker={'size': 15})],
        'layout': go.Layout(
            xaxis={'title': 'Petal Length (cm)'},
            yaxis={'title': 'Petal Width (cm)'},
            title='Petal Length/Width Correlation for {}'.format(iris_class)
        )
    }
    return chart


@app.callback(Output('table1', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('dropdown-state', 'value')])
def dropdown_filter(clicks, dropdown_value):
    dff = df[df.iris_class == dropdown_value].iloc[:, :2]
    table_header = [html.Tr([html.Th(col) for col in dff.columns])]
    table_body = [html.Tr([html.Td(dff.iloc[i][col]) for col in dff.columns])
                  for i in range(len(dff))]
    return table_header + table_body


if __name__ == '__main__':
    app.run_server(debug=True)