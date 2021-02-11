import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from fastai.learner import load_learner


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
            'width': '100%',
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
            'textAlign': 'center'
        }
    ),
])



@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output(content):
    if content is not None:
        img_str = base64.b64decode(content.split(',')[-1])
        img_io = BytesIO(img_str)
        img = Image.open(img_io)

        print(learn.predict(np.array(img)))

        return html.Img(src=content)


if __name__ == '__main__':
    app.run_server(debug=True)