import dash
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

import glob
import pandas as pd

from dash.dependencies import Input, Output
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--log-dir', required=True, type=str, help='Log directory')
parser.add_argument('--interval', required=False, type=int, default=5, help='Update interval')
args = vars(parser.parse_args())
csv_files = glob.glob(f'{args["log_dir"]}/*.csv')
graph_names = [x.split('/')[-1].split('.')[0] for x in csv_files]
log_name = args['log_dir'].split('/')[-1]
subplot_titles = [f'{log_name}_{x}' for x in graph_names]

app = dash.Dash(__name__)
layout = [
    dcc.Interval(id='graph-update', interval=1000*args['interval']),
    dcc.Graph(id='graph', animate=True)
]
app.layout = html.Div(layout)

@app.callback(Output('graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph(n):
    data = []
    fig = make_subplots(rows=len(graph_names), 
            cols=1,
            vertical_spacing=0.1,
            subplot_titles=subplot_titles)
    for csv_file in csv_files:
        data.append(pd.read_csv(csv_file))

    for i, d in enumerate(data):
        t = list(range(len(d)))
        max_y = 0
    
        for col in list(d.columns[1:]):
            fig.add_trace(
                go.Scatter(
                    x = t,
                    y = d[col].values,
                    name = col,
                    mode = 'lines+markers'
                ),
                row = i + 1,
                col = 1
            )

    
    for i in range(len(data)):
        fig.update_xaxes(range=[-1, len(data[i])], row=i+1, col=1)
    fig.update_layout(height=600*len(data), title_text=log_name)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
