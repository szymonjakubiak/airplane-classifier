import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from fastai.learner import load_learner
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

learn = load_learner('resnet50_v2.pkl')


app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100pv',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload',
        style={
            'textAlign': 'left',
            'width': '20%'
        }
    ),
    html.Div(id='predictions-graph',
        style={
            'textAlign': 'right',
            'width': '70%'
        }
    )
])



@app.callback(Output('output-image-upload', 'children'),
              Output('predictions-graph', 'children'),
              Input('upload-image', 'contents'))
def update_output(content):
    if content is not None:
        # decode base64 string
        padding, encoded_img = content.split(',')
        img_str = base64.b64decode(encoded_img)
        img_io = BytesIO(img_str)
        img = Image.open(img_io)

        # print(learn.predict(np.array(img)))

        # resize and get b64 string
        img_thumb = img.resize(size=(224, 224))
        img_buffer = BytesIO()
        img_thumb.save(img_buffer, 'png')
        img_thumb_str = padding + ',' + base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        vocab = learn.dls.vocab
        predicted_class, _, predictions = learn.predict(np.array(img))


        response_block_image = html.Div([
            html.Div(str(learn.predict(np.array(img)))),
            html.Img(src=img_thumb_str)
        ])



        df = pd.DataFrame.from_dict({'label': vocab, 'prediction': predictions}).sort_values('prediction')
        fig = px.bar(df, x='prediction', y='label')

        response_block_graph = html.Div([
            dcc.Graph(figure=fig)
        ])



        return response_block_image, response_block_graph
    return None, None


if __name__ == '__main__':
    app.run_server(debug=True)