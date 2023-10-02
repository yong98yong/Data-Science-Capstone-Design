import flask
import datetime
import dash
from dash import Dash
from dash import dcc
from dash import html
from dash.html import Br
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_daq as daq
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

application = flask.Flask(__name__)


dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX], server=application, url_base_pathname='/dashapp/',suppress_callback_exceptions=True)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# mysql에서 데이터 불러오기
def load_data():
    global df, df_bio, df_temp, df_hum, scaler1
    db_connection_str = 'mysql+pymysql://root:root@ec2-52-87-174-86.compute-1.amazonaws.com:3306/capstone_db'
    # db_connection_str = 'mysql+pymysql://root:23599532@127.0.0.1:3306/capstone_db'
    db_connection = create_engine(db_connection_str)
    db_conn = db_connection.connect()
    df = pd.read_sql_query("SELECT * FROM packets_db ORDER BY idpackets_db DESC LIMIT 200", db_conn)
    df = df.sort_values('idpackets_db')
    # df = pd.read_sql_table('packets_db', db_conn)
    db_conn.close()
    df_ff=df[df['Sensor ID']=='ff']
    packet_datafield_event=[]
    packet_datafield_g=[]
    for i in df_ff[df_ff['CMD']=='80'].index:
        if bytes.fromhex(df_ff[df_ff['CMD']=='80']['Data'][i]).decode("ascii")[1]=='E':
            packet_datafield_event.append(bytes.fromhex(df_ff[df_ff['CMD']=='80']['Data'][i]).decode("ascii"))
        else:
            packet_datafield_g.append(bytes.fromhex(df_ff[df_ff['CMD']=='80']['Data'][i]).decode("ascii"))

    bio=[]
    temp=[]
    hum=[]
    for i in range(len(packet_datafield_event)):
        packet_datafield_event[i]=packet_datafield_event[i][19::][0:-26]
        if packet_datafield_event[i][11:24]=='Recv Bio Data':
            bio.append(packet_datafield_event[i])
        elif packet_datafield_event[i][11:17]=='Read T':
            temp.append(packet_datafield_event[i])
        elif packet_datafield_event[i][11:17]=='Read H':
            hum.append(packet_datafield_event[i])

    df_bio=pd.DataFrame([bio[i].split(' ') for i in range(len(bio))])
    df_temp=pd.DataFrame([temp[i].split(' ') for i in range(len(temp))])
    df_hum=pd.DataFrame([hum[i].split(' ') for i in range(len(hum))])

    df_bio['Sensor']=df_bio[1]+' '+ df_bio[2]+' '+df_bio[3]
    df_bio.drop(columns=[1, 2, 3], inplace=True)
    df_bio.columns=['Time', 'PD', 'ACT', 'BR', 'HR', 'X', 'Y', 'Sensor']

    df_temp['Sensor']=df_temp[1]+' '+ df_temp[2]+' '+df_temp[3]
    df_temp.drop(columns=[1, 2, 3], inplace=True)
    df_temp.columns=['Time', 'Temp', 'Sensor']

    df_hum['Sensor']=df_hum[1]+' '+ df_hum[2]+' '+df_hum[3]
    df_hum.drop(columns=[1, 2, 3], inplace=True)
    df_hum.columns=['Time', 'Hum', 'Sensor']

    for i in df_bio.columns[1:7]:
        df_bio[i]=df_bio[i].str.extract(r'(\d+)')
    df_bio['Time'] = df_bio['Time'].str.strip('[]')
        
    df_temp['Temp']=df_temp['Temp'].str.extract(r'(\d+.\d)')
    df_temp['Time'] = df_temp['Time'].str.strip('[]')

    df_hum['Hum']=df_hum['Hum'].str.extract(r'(\d+.\d)')
    df_hum['Time'] = df_hum['Time'].str.strip('[]')
        
    df_bio[['PD', 'ACT', 'BR', 'HR', 'X', 'Y']]=df_bio[['PD', 'ACT', 'BR', 'HR', 'X', 'Y']].apply(pd.to_numeric)
    df_temp['Temp']=pd.to_numeric(df_temp['Temp'])
    df_hum['Hum']=pd.to_numeric(df_hum['Hum'])

    
    df_bio=pd.read_csv('data/df_bio_test.csv',index_col = 0)
    df_bio[['PD', 'ACT', 'BR', 'HR', 'X', 'Y']]=df_bio[['PD', 'ACT', 'BR', 'HR', 'X', 'Y']].apply(pd.to_numeric)
load_data()

def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("DEMO DASHBOARD"),
                    html.H6("Capstone Project"),
                ]
            )
        ]
    )


def generate_modal():
    return html.Div(
        id="markdown",
        className="modal",
        children=(
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=dcc.Markdown(
                            children=(
                                """
                        ###### 출처
                        [Github repository](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard).
                    """
                            )
                        ),
                    ),
                ],
            )
        ),
    )

def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="설정",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="차트 대시보드",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )

def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

def generate_metric_row(id, style, col1, col2, col3):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"margin-right": "2.5rem", "minWidth": "50px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"height": "100%"},   
                className="six columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"display": "flex", "justifyContent": "center"},
                className="one column",
                children=col3["children"],
            )
        ],
    )

def generate_metric_list_header():
    return generate_metric_row(
        "metric_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("종류")},
        {"id": "m_header_2", "children": html.Div("분포")},
        {"id": "m_header_3", "children": html.Div("상태")},
    )

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_indicator = "_indicator"

def generate_metric_row_helper(df,index):
    div_id = df.columns[index] + suffix_row
    button_id = df.columns[index] + suffix_button_id
    sparkline_graph_id = df.columns[index] + suffix_sparkline_graph
    indicator_id = df.columns[index] + suffix_indicator

    return generate_metric_row(
        div_id,
        None,
        {
            "id": df.columns[index],
            "className": "metric-row-button-text",
            "children": html.Button(
                id=button_id,
                className="metric-row-button",
                children=df.columns[index],
                title="Click to visualize live chart",
                n_clicks=0,
            ),
        },
        {
            "id": df.columns[index] + "_sparkline",
            "children": dcc.Graph(
                id=sparkline_graph_id,
                style={"width": "100%", "height": "95%"},
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df['Time'],
                                "y": df[df.columns[index]],
                                "mode": "lines",
                                "name": df.columns[index],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                ),
            ),
        },
        {
            "id": df.columns[index] + "_pf",
            "children": daq.Indicator(
                id=indicator_id, value=True, color="#91dfd2", size=12
            ),
        }
    )

def build_top_panel():
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("Summary"),
                    html.Div(
                        id="metric-div",
                        children=[
                            generate_metric_list_header(),
                            html.Div(
                                id="metric-rows",
                                children=[
                                    generate_metric_row_helper(df_bio,3),
                                    generate_metric_row_helper(df_bio,4),
                                    generate_metric_row_helper(df_bio,2),
                                    generate_metric_row_helper(df_temp,1),
                                    generate_metric_row_helper(df_hum,1),
                                    generate_metric_row_helper(df_bio,1),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


out = '외출'
def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="card-1",
                children=[
                    html.Div(className="section-banner2", children='응급현황'),
                    html.Img(id="h", src=dash_app.get_asset_url("h.png")
                    ),
                    html.Div(className="section-banner2", children='외출여부'),
                    html.Img(id="home", src=dash_app.get_asset_url("home.png")
                    )
                   ],
            )
        ],
    )

def build_chart_panel():
    return html.Div(
        id="control-chart-container",
        className="twelve columns",
        children=[
            generate_section_banner("Live Chart"),
            dcc.Graph(
                id="control-chart-live",
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": [],
                                "y": [],
                                "mode": "lines",
                                "name":df_temp.columns[1],   
                            }
                        ],
                        "layout": {
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                            "xaxis": dict(
                                showline=False, showgrid=False, zeroline=False
                            ),
                            "yaxis": dict(
                                showgrid=False, showline=False, zeroline=False
                            ),
                            "autosize": True,
                        },
                    }
                ),
            ),
        ],
    )

def make_data(data, window_size, column):
    feature_list =[] 
    for i in range(len(data)-window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size][column]))            
    data_X = np.array(feature_list)        
        
    return data_X

def flatten(lst):
    result=[]
    for i in lst:
        result.extend(i)
    return(result)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 5, 3) ## 1*30-> 5*28
        self.relu1 = nn.ReLU()
        self.max1d1 = nn.MaxPool1d(2, stride =2) ## 5*28 -> 5*14
        self.conv2 = nn.Conv1d(5,10,3) ## 5*14->10*12
        self.relu2 = nn.ReLU()
        self.max1d2 = nn.MaxPool1d(2, stride = 2) ##10*12->10*6
        
        self.fc1 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 1)
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1d1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max1d2(x)
        x= x.view(-1, 60)
        
        x= self.fc1(x)
        x= self.fc2(x)
        x= self.fc3(x)
       
        return x

model_br = torch.load("model/model_br.pt")
model_hr = torch.load("model_hr.pt")

def generate_graph(df, interval, specs_dict, col):
    col_data = df[col]
    mean = df[col].mean()
    ucl = specs_dict[col]["ucl"]
    lcl = specs_dict[col]["lcl"]
    usl = specs_dict[col]["usl"]
    lsl = specs_dict[col]["lsl"]

    x_array = df['Time']
    y_array = col_data
    ooc_trace = {
        "x": [],
        "y": [],
        "name": "Outlier",
        "mode": "markers",
        "marker": dict(color="rgba(210, 77, 87, 0.7)", symbol="square", size=11),
    }

    ul_br = 21.56016272560573
    ll_br = -18.8437187004669
    ul_hr=121.449771539840190
    ll_hr=-72.6413671604271
    df_bio2 = df_bio[df_bio['PD']==1]
    
    if col in ['Temp', 'Hum']:    
        for index, data in zip(x_array,y_array):
                if (data >= ucl or data <= lcl):
                    ooc_trace["x"].append(index)
                    ooc_trace["y"].append(data)

    if (len(df_bio2)>30):
        if col =='BR':
            X=make_data(df_bio2, 30, col)
            pred=[]
            for i in range(len(X)):
                sc=MinMaxScaler()
                globals()['X{}'.format(i)]=sc.fit_transform(np.array(X[i]).reshape(-1,1))
                pred0=torch.sigmoid(model_br(torch.FloatTensor(globals()['X{}'.format(i)]).view(1, 30)))
                pred0=sc.inverse_transform(np.array(flatten(pred0.tolist())).reshape(-1,1))
                pred.append(flatten(pred0))
            original=df_bio2[col].iloc[30:]
            original_time = df_bio2['Time'].iloc[30:].to_list()
            inv_pred=np.array(pred).reshape(-1,)
            error_br=pd.DataFrame({'Test': original, 'Predction': inv_pred, 'Error': original-inv_pred})
            error_br.index = range(0, len(error_br))
            for i in error_br[(error_br['Error']<=ll_br) | (error_br['Error']>=ul_br) | (error_br['Test']<1)].index:
                ooc_trace["x"].append(original_time[i])
                ooc_trace["y"].append(df_bio2['BR'].iloc[30:].iloc[i])
        
        elif col =='HR':
            X=make_data(df_bio2, 30, col)
            pred=[]
            for i in range(len(X)):
                sc=MinMaxScaler()
                globals()['X{}'.format(i)]=sc.fit_transform(np.array(X[i]).reshape(-1,1))
                pred0=torch.sigmoid(model_hr(torch.FloatTensor(globals()['X{}'.format(i)]).view(1, 30)))
                pred0=sc.inverse_transform(np.array(flatten(pred0.tolist())).reshape(-1,1))
                pred.append(flatten(pred0))
            original=df_bio2[col].iloc[30:]
            original_time = df_bio2['Time'].iloc[30:].to_list()
            inv_pred=np.array(pred).reshape(-1,)
            error_hr=pd.DataFrame({'Test': original, 'Predction': inv_pred, 'Error': original-inv_pred})
            error_hr.index = range(0, len(error_hr))
            for i in error_hr[(error_hr['Error']<=ll_hr) | (error_hr['Error']>=ul_hr) | (error_hr['Test']<1)].index:
                ooc_trace["x"].append(original_time[i])
                ooc_trace["y"].append(df_bio2['HR'].iloc[30:].iloc[i])
         
    histo_trace = {
        "x": x_array,
        "y": y_array,
        "type": "histogram",
        "orientation": "h",
        "name": "Distribution",
        "xaxis": "x2",
        "yaxis": "y2",
        "marker": {"color": "#f4d44d"},
    }

    fig = {
        "data": [
            {
                "x": x_array,
                "y": y_array,
                "mode": "lines",
                "name": col,
                "line": {"color": "#f4d44d"},

            },
            ooc_trace,
            histo_trace,
        ]
    }

    len_figure = len(fig["data"][0]["x"])

    fig["layout"] = dict(
        margin=dict(t=40),
        hovermode="closest",
        uirevision=col,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.1},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "zeroline": False,
            "showgrid": False,
            "title": "date",
            "showline": False,
            "domain": [0, 0.78],
            "titlefont": {"color": "darkgray"},
        },
        yaxis={
            "title": col,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
            "titlefont": {"color": "darkgray"},
        
        },
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": usl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": usl,
                "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": lsl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": lsl,
                "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": ucl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": ucl,
                "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": mean,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": mean,
                "line": {"color": "rgb(255,127,80)", "width": 2},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": lcl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": lcl,
                "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
            },
        ],
        xaxis2={
            "title": "Count",
            "domain": [0.82, 1],  # 70 to 100 % of width
            "titlefont": {"color": "darkgray"},
            "showgrid": False,
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
            "titlefont": {"color": "darkgray"},
        },
    )

    return fig

def generate_graph_act(df, interval, specs_dict, col):
    col_data = df[col]
    mean = df[col].mean()
    ucl = specs_dict[col]["ucl"]
    lcl = specs_dict[col]["lcl"]
    usl = specs_dict[col]["usl"]
    lsl = specs_dict[col]["lsl"]

    x_array = df['Time']
    y_array = col_data

    ooc_trace = {
        "x": [],
        "y": [],
        "name": "Outlier",
        "mode": "markers",
        "marker": dict(color="rgba(210, 77, 87, 0.7)", symbol="square", size=11),
    }

    for index, data in zip(x_array,y_array):
        if ((data <= lcl) & (int(df['PD'][df['Time']==index])==1)):
            ooc_trace["x"].append(index)
            ooc_trace["y"].append(data)

    histo_trace = {
        "x": x_array,
        "y": y_array,
        "type": "histogram",
        "orientation": "h",
        "name": "Distribution",
        "xaxis": "x2",
        "yaxis": "y2",
        "marker": {"color": "#f4d44d"},
    }

    fig = {
        "data": [
            {
                "x": x_array,
                "y": y_array,
                "mode": "lines",
                "name": col,
                "line": {"color": "#f4d44d"},

            },
            ooc_trace,
            histo_trace,
        ]
    }

    len_figure = len(fig["data"][0]["x"])

    fig["layout"] = dict(
        margin=dict(t=40),
        hovermode="closest",
        uirevision=col,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.1},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "zeroline": False,
            "showgrid": False,
            "title": "date",
            "showline": False,
            "domain": [0, 0.78],
            "titlefont": {"color": "darkgray"},
        },
        yaxis={
            "title": col,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
            "titlefont": {"color": "darkgray"},
        },
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": usl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": usl,
                "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": lsl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": lsl,
                "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": ucl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": ucl,
                "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": mean,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": mean,
                "line": {"color": "rgb(255,127,80)", "width": 2},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": df['Time'].to_list()[0],
                "y0": lcl,
                "x1": df['Time'].to_list()[len(df['Time'])-1],
                "y1": lcl,
                "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
            },
        ],
        xaxis2={
            "title": "Count",
            "domain": [0.82, 1],  # 70 to 100 % of width
            "titlefont": {"color": "darkgray"},
            "showgrid": False,
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
            "titlefont": {"color": "darkgray"},
        },
    )

    return fig

def generate_graph_pd(df, interval, specs_dict, col):
    col_data = df[col]
    mean = df[col].mean()
    ucl = specs_dict[col]["ucl"]
    lcl = specs_dict[col]["lcl"]
    usl = specs_dict[col]["usl"]
    lsl = specs_dict[col]["lsl"]

    x_array = df['Time']
    y_array = col_data

    ooc_trace = {
        "x": [],
        "y": [],
        "name": "Outlier",
        "mode": "markers",
        "marker": dict(color="rgba(210, 77, 87, 0.7)", symbol="square", size=11),
    }

    histo_trace = {
        "x": x_array,
        "y": y_array,
        "type": "histogram",
        "orientation": "h",
        "name": "Distribution",
        "xaxis": "x2",
        "yaxis": "y2",
        "marker": {"color": "#f4d44d"},
    }

    fig = {
        "data": [
            {
                "x": x_array,
                "y": y_array,
                "mode": "lines",
                "name": col,
                "line": {"color": "#f4d44d"},

            },
            ooc_trace,
            histo_trace,
        ]
    }

    len_figure = len(fig["data"][0]["x"])

    fig["layout"] = dict(
        margin=dict(t=40),
        hovermode="closest",
        uirevision=col,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.1},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "zeroline": False,
            "showgrid": False,
            "title": "date",
            "showline": False,
            "domain": [0, 0.78],
            "titlefont": {"color": "darkgray"},
        },
        yaxis={
            "title": col,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
            "titlefont": {"color": "darkgray"},
        },
        # shapes=[
        #     {
        #         "type": "line",
        #         "xref": "x",
        #         "yref": "y",
        #         "x0": df['Time'].to_list()[0],
        #         "y0": usl,
        #         "x1": df['Time'].to_list()[len(df['Time'])-1],
        #         "y1": usl,
        #         "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
        #     },
        #     {
        #         "type": "line",
        #         "xref": "x",
        #         "yref": "y",
        #         "x0": df['Time'].to_list()[0],
        #         "y0": lsl,
        #         "x1": df['Time'].to_list()[len(df['Time'])-1],
        #         "y1": lsl,
        #         "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
        #     },
        #     {
        #         "type": "line",
        #         "xref": "x",
        #         "yref": "y",
        #         "x0": df['Time'].to_list()[0],
        #         "y0": ucl,
        #         "x1": df['Time'].to_list()[len(df['Time'])-1],
        #         "y1": ucl,
        #         "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
        #     },
        #     {
        #         "type": "line",
        #         "xref": "x",
        #         "yref": "y",
        #         "x0": df['Time'].to_list()[0],
        #         "y0": mean,
        #         "x1": df['Time'].to_list()[len(df['Time'])-1],
        #         "y1": mean,
        #         "line": {"color": "rgb(255,127,80)", "width": 2},
        #     },
        #     {
        #         "type": "line",
        #         "xref": "x",
        #         "yref": "y",
        #         "x0": df['Time'].to_list()[0],
        #         "y0": lcl,
        #         "x1": df['Time'].to_list()[len(df['Time'])-1],
        #         "y1": lcl,
        #         "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
        #     },
        # ],
        xaxis2={
            "title": "Count",
            "domain": [0.82, 1],  # 70 to 100 % of width
            "titlefont": {"color": "darkgray"},
            "showgrid": False,
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
            "titlefont": {"color": "darkgray"},
        },
    )

    return fig

def init_df():
    ret = {}
    for col in df_temp.columns[1:2]:
        data = df_temp[col]
        stats = data.describe()
        std = stats["std"].tolist()
        ucl = 28
        lcl = 18
        usl = 26 
        lsl = 20
        
        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                }
            }
        )
    for col in df_hum.columns[1:2]:
        data = df_hum[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = 60.1
        lcl = 39.9
        usl = 60.1
        lsl = 39.9

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                }
            }
        )

    for col in ['ACT']:
        data = df_bio[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = (stats["max"] + 1).tolist()
        lcl = 0
        usl = (stats["max"] + 1).tolist()
        lsl = 0

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                }
            }
        )
    
    for col in ['BR']:
        data = df_bio[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = 0
        lcl = 0
        usl = 0
        lsl = 0

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                }
            }
        )

    for col in ['HR']:
        data = df_bio[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = 0
        lcl = 0
        usl = 0
        lsl = 0

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                }
            }
        )

    for col in ['PD']:
        data = df_bio[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = (stats["mean"] + 3 * stats["std"]).tolist()
        lcl = (stats["mean"] - 3 * stats["std"]).tolist()
        usl = (stats["mean"] + stats["std"]).tolist()
        lsl = (stats["mean"] - stats["std"]).tolist()

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                }
            }
        )
    
    return ret

state_dict = init_df()

ud_usl_input = daq.NumericInput(
id="ud_usl_input",className="setting-input", size=200, max=9999999
)
ud_lsl_input = daq.NumericInput(
id="ud_lsl_input", className="setting-input", size=200, max=9999999
)
ud_ucl_input = daq.NumericInput(
id="ud_ucl_input", className="setting-input", size=200, max=9999999
)
ud_lcl_input = daq.NumericInput(
id="ud_lcl_input", className="setting-input", size=200, max=9999999
)

def init_value_setter_store():
    # Initialize store data
    global state_dict
    state_dict = init_df()
    return state_dict

def build_tab_1():
    return [
        # Manually select metrics
        html.Div(
            id="set-specs-intro-container",
            # className='twelve columns',
            children=html.P(
                "이상치 기준 설정"
            ),
        ),
        html.Div(
            id="settings-menu",
            children=[
                html.Div(
                    id="metric-select-menu",
                    # className='five columns',
                    children=[
                        html.Label(id="metric-select-title", children="종류 선택"),
                        html.Br(),
                        dcc.Dropdown(
                            id="metric-select-dropdown",
                            options=list(
                                [{"label": param, "value": param} for param in df_bio.columns[1:5].tolist()+df_temp.columns[1:2].tolist()+df_hum.columns[1:2].tolist()]
                            ),
                            value=df_temp.columns[1],
                        ),
                    ],
                ),
                html.Div(
                    id="value-setter-menu",
                    # className='six columns',
                    children=[
                        html.Div(id="value-setter-panel",
                        children=[
                            html.Div([daq.NumericInput(id="ud_usl_input",className="setting-input", size=200, max=9999999),
                            daq.NumericInput(id="ud_lsl_input", className="setting-input", size=200, max=9999999),
                            daq.NumericInput(id="ud_ucl_input", className="setting-input", size=200, max=9999999),
                            daq.NumericInput(id="ud_lcl_input", className="setting-input", size=200, max=9999999)
                            ])
                        ]),
                        html.Br(),
                        html.Div(
                            id="button-div",
                            children=[
                                html.Button("Update", id="value-setter-set-btn"),
                                html.Button(
                                    "View current setup",
                                    id="value-setter-view-btn",
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        html.Div(
                            id="value-setter-view-output", className="output-datatable"
                        ),
                    ],
                ),
            ],
        ),
    ]

def build_value_setter_line(line_num, label, value, col3):
    return html.Div(
        id=line_num,
        children=[
            html.Label(label, className="four columns"),
            html.Label(value, className="four columns"),
            html.Div(col3, className="four columns"),
        ],
        className="row",
    )

dash_app.layout = html.Div(
    id="big-app-container",    
    children=[ 
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval= 5 * 1000,  # in milliseconds
            n_intervals=0
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content")
            ],
        ),
    dcc.Store(id="value-setter-store", data=init_value_setter_store()),
    generate_modal()]
)

@dash_app.callback(
    Output("app-content", "children"),
    Input("app-tabs", "value"),
    State('interval-component', 'n_intervals')
)
def render_tab_content(tab_switch,interval):
    if tab_switch == "tab1":
        return build_tab_1()
    return (
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),            
                html.Div(
                    id="graphs-container",
                    children=[build_top_panel(),build_chart_panel()]
                )
            ]
        )
    )

# @dash_app.callback(
#     Output("markdown", "style"),
#     [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
# )
# def update_click_output(button_click, close_click):
#     ctx = dash.callback_context

#     if ctx.triggered:
#         prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
#         if prop_id == "learn-more-button":
#             return {"display": "block"}

#     return {"display": "none"}

outlier_list = [-1,-1,-1,-1,-1,-1]
@dash_app.callback(
    output=Output("control-chart-live", "figure"),
    inputs=[
        Input("interval-component", "n_intervals"),
        Input(df_temp.columns[1] + suffix_button_id, "n_clicks"),
        Input(df_hum.columns[1] + suffix_button_id, "n_clicks"),
        Input(df_bio.columns[1] + suffix_button_id, "n_clicks"),
        Input(df_bio.columns[2] + suffix_button_id, "n_clicks"),
        Input(df_bio.columns[3] + suffix_button_id, "n_clicks"),
        Input(df_bio.columns[4] + suffix_button_id, "n_clicks")
    ],
    state= [State("value-setter-store", "data"),State("control-chart-live", "figure")]
)
def update_control_chart(interval, n1, n2, n3, n4, n5, n6, data, cur_fig):
    data = init_df()
    global outlier_list
    fig1 = generate_graph(df_temp, interval, data, df_temp.columns[1])
    if ((len(fig1["data"][1]['x'])>0) & (outlier_list[0]==-1)):
        outlier_list[0]=outlier_list[0]*-1
    elif ((len(fig1["data"][1]['x'])==0) & (outlier_list[0]==1)):
        outlier_list[0]=outlier_list[0]*-1
    
    fig2 = generate_graph(df_hum, interval, data, df_hum.columns[1])
    if ((len(fig2["data"][1]['x'])>0) & (outlier_list[1]==-1)):
        outlier_list[1]=outlier_list[1]*-1
    elif ((len(fig2["data"][1]['x'])==0) & (outlier_list[1]==1)):
        outlier_list[1]=outlier_list[1]*-1

    fig3 = generate_graph_pd(df_bio, interval, data, df_bio.columns[1])
    if ((len(fig3["data"][1]['x'])>0) & (outlier_list[2]==-1)):
        outlier_list[2]=outlier_list[2]*-1
    elif ((len(fig3["data"][1]['x'])==0) & (outlier_list[2]==1)):
        outlier_list[2]=outlier_list[2]*-1

    fig4 = generate_graph_act(df_bio, interval, data, df_bio.columns[2])
    if ((len(fig4["data"][1]['x'])>0) & (outlier_list[3]==-1)):
        outlier_list[3]=outlier_list[3]*-1
    elif ((len(fig4["data"][1]['x'])==0) & (outlier_list[3]==1)):
        outlier_list[3]=outlier_list[3]*-1

    fig5 = generate_graph(df_bio, interval, data, df_bio.columns[3])
    if ((len(fig5["data"][1]['x'])>0) & (outlier_list[4]==-1)):
        outlier_list[4]=outlier_list[4]*-1
    elif ((len(fig5["data"][1]['x'])==0) & (outlier_list[4]==1)):
        outlier_list[4]=outlier_list[4]*-1

    fig6 = generate_graph(df_bio, interval, data, df_bio.columns[4])
    if ((len(fig6["data"][1]['x'])>0) & (outlier_list[5]==-1)):
        outlier_list[5]=outlier_list[5]*-1
    elif ((len(fig6["data"][1]['x'])==0) & (outlier_list[5]==1)):
        outlier_list[5]=outlier_list[5]*-1

    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return generate_graph(df_bio, interval, data, df_bio.columns[3])

    if ctx.triggered:
        # Get most recently triggered id and prop_type
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
        if prop_type == "n_clicks":
            curr_id = cur_fig["data"][0]["name"]
            prop_id = prop_id[:-7]
            if (curr_id == prop_id) & (prop_id == 'Temp'):
                return generate_graph(df_temp,interval, data, curr_id)
            elif (curr_id == prop_id) & (prop_id == 'Hum'):
                return generate_graph(df_hum,interval, data, curr_id)
            elif (curr_id == prop_id) & (prop_id == 'PD'):
                return generate_graph_pd(df_bio,interval, data, curr_id)
            elif (curr_id == prop_id) & (prop_id == 'ACT'):
                return generate_graph_act(df_bio,interval, data, curr_id)
            elif (curr_id == prop_id) & (prop_id not in ['Temp','Hum','PD','ACT']):
                return generate_graph(df_bio,interval, data, curr_id)
            elif (curr_id != prop_id) & (prop_id == 'Temp'):
                return generate_graph(df_temp,interval, data, prop_id)
            elif (curr_id != prop_id) & (prop_id == 'Hum'):
                return generate_graph(df_hum,interval, data, prop_id)
            elif (curr_id != prop_id) & (prop_id == 'PD'):
                return generate_graph_pd(df_bio,interval, data, prop_id)
            elif (curr_id != prop_id) & (prop_id == 'ACT'):
                return generate_graph_act(df_bio,interval, data, prop_id)        
            else:
                return generate_graph(df_bio,interval, data, prop_id)

        if prop_type == "n_intervals" and cur_fig is not None:
            curr_id = cur_fig["data"][0]["name"]
            if curr_id =='Temp':
                return generate_graph(df_temp, interval, data, curr_id)
            elif curr_id =='Hum':
                return generate_graph(df_hum, interval, data, curr_id)
            elif curr_id =='PD':
                return generate_graph_pd(df_bio, interval, data, curr_id)
            elif curr_id =='ACT':
                return generate_graph_act(df_bio, interval, data, curr_id)
            else:
                return generate_graph(df_bio, interval, data, curr_id)

@dash_app.callback(
    output = Output(df_temp.columns[1] + suffix_sparkline_graph, "figure"),
    inputs = Input("interval-component", "n_intervals"),
)
def update_sparkline(interval):
    load_data()
    state_dict = init_df()
    figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df_temp['Time'],
                                "y": df_temp[df_temp.columns[1]],
                                "mode": "lines",
                                "name": df_temp.columns[1],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                tickfont = dict(
                                    family = 'Old Standard TT, serif',
                                    size = 10,
                                    color = 'darkgray'
                                    )                                
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                )
    return figure

@dash_app.callback(
    output = Output(df_hum.columns[1] + suffix_sparkline_graph, "figure"),
    inputs = Input("interval-component", "n_intervals"),
)
def update_sparkline(interval):
    figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df_hum['Time'],
                                "y": df_hum[df_hum.columns[1]],
                                "mode": "lines",
                                "name": df_hum.columns[1],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                tickfont = dict(
                                    family = 'Old Standard TT, serif',
                                    size = 10,
                                    color = 'darkgray'
                                    )                                
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                )
    return figure

@dash_app.callback(
    output = Output(df_bio.columns[1] + suffix_sparkline_graph, "figure"),
    inputs = Input("interval-component", "n_intervals"),
)
def update_sparkline(interval):
    figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df_bio['Time'],
                                "y": df_bio[df_bio.columns[1]],
                                "mode": "lines",
                                "name": df_bio.columns[1],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                tickfont = dict(
                                    family = 'Old Standard TT, serif',
                                    size = 10,
                                    color = 'darkgray'
                                    )                                
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                )
    return figure

@dash_app.callback(
    output = Output(df_bio.columns[2] + suffix_sparkline_graph, "figure"),
    inputs = Input("interval-component", "n_intervals"),
)
def update_sparkline(interval):
    figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df_bio['Time'],
                                "y": df_bio[df_bio.columns[2]],
                                "mode": "lines",
                                "name": df_bio.columns[2],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                             "yaxis": dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                tickfont = dict(
                                    family = 'Old Standard TT, serif',
                                    size = 10,
                                    color = 'darkgray'
                                    )                                
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                )
    return figure

@dash_app.callback(
    output = Output(df_bio.columns[3] + suffix_sparkline_graph, "figure"),
    inputs = Input("interval-component", "n_intervals"),
)
def update_sparkline(interval):
    figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df_bio['Time'],
                                "y": df_bio[df_bio.columns[3]],
                                "mode": "lines",
                                "name": df_bio.columns[3],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                tickfont = dict(
                                    family = 'Old Standard TT, serif',
                                    size = 10,
                                    color = 'darkgray'
                                    )                                
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                )
    return figure

@dash_app.callback(
    output = Output(df_bio.columns[4] + suffix_sparkline_graph, "figure"),
    inputs = Input("interval-component", "n_intervals"),
)
def update_sparkline(interval):
    figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": df_bio['Time'],
                                "y": df_bio[df_bio.columns[4]],
                                "mode": "lines",
                                "name": df_bio.columns[4],
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                tickfont = dict(
                                    family = 'Old Standard TT, serif',
                                    size = 10,
                                    color = 'darkgray'
                                    )                                
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    },
                )
    return figure

@dash_app.callback(
    output=[
        Output("value-setter-panel", "children"),
        Output("ud_usl_input", "value"),
        Output("ud_lsl_input", "value"),
        Output("ud_ucl_input", "value"),
        Output("ud_lcl_input", "value"),
    ],
    inputs=[Input("metric-select-dropdown", "value")],
    state=[State("value-setter-store", "data")],
)
def build_value_setter_panel(dd_select, state_value):
    return (
        [
            build_value_setter_line(
                "value-setter-panel-header",
                "Specs",
                "Historical Value",
                "Set new value",
            ),
            build_value_setter_line(
                "value-setter-panel-usl",
                "Upper Specification limit",
                state_dict[dd_select]["usl"],
                ud_usl_input,
            ),
            build_value_setter_line(
                "value-setter-panel-lsl",
                "Lower Specification limit",
                state_dict[dd_select]["lsl"],
                ud_lsl_input,
            ),
            build_value_setter_line(
                "value-setter-panel-ucl",
                "Upper Control limit",
                state_dict[dd_select]["ucl"],
                ud_ucl_input,
            ),
            build_value_setter_line(
                "value-setter-panel-lcl",
                "Lower Control limit",
                state_dict[dd_select]["lcl"],
                ud_lcl_input,
            ),
        ],
        state_value[dd_select]["usl"],
        state_value[dd_select]["lsl"],
        state_value[dd_select]["ucl"],
        state_value[dd_select]["lcl"],
    )


# @dash_app.callback(
#     Output("value-setter-store", "data"),
#     Input("interval-component", "n_intervals")
# )
# def data_refresh(Interval):
#     return init_value_setter_store()

@dash_app.callback(
    output=Output("value-setter-store", "data"),
    inputs=[Input("value-setter-set-btn", "n_clicks")],
    state=[
        State("metric-select-dropdown", "value"),
        State("value-setter-store", "data"),
        State("ud_usl_input", "value"),
        State("ud_lsl_input", "value"),
        State("ud_ucl_input", "value"),
        State("ud_lcl_input", "value"),
    ],
)
def set_value_setter_store(set_btn, param, interval, data, usl, lsl, ucl, lcl):

    if set_btn is None:
        return data
    else:
        data[param]["usl"] = usl
        data[param]["lsl"] = lsl
        data[param]["ucl"] = ucl
        data[param]["lcl"] = lcl

        # Recalculate ooc in case of param updates
        return data


@dash_app.callback(
    output=Output("value-setter-view-output", "children"),
    inputs=[
        Input("value-setter-view-btn", "n_clicks"),
        Input("metric-select-dropdown", "value"),
        Input("value-setter-store", "data"),
    ],
)
def show_current_specs(n_clicks, dd_select, store_data):
    if n_clicks > 0:
        curr_col_data = store_data[dd_select]
        new_df_dict = {
            "Specs": [
                "Upper Specification Limit",
                "Lower Specification Limit",
                "Upper Control Limit",
                "Lower Control Limit",
            ],
            "Current Setup": [
                curr_col_data["usl"],
                curr_col_data["lsl"],
                curr_col_data["ucl"],
                curr_col_data["lcl"],
            ],
        }
        new_df = pd.DataFrame.from_dict(new_df_dict)
        return dash_table.DataTable(
            style_header={"fontWeight": "bold", "color": "inherit"},
            style_as_list_view=True,
            fill_width=True,
            style_cell_conditional=[
                {"if": {"column_id": "Specs"}, "textAlign": "left"}
            ],
            style_cell={
                "backgroundColor": "#1e2130",
                "fontFamily": "Open Sans",
                "padding": "0 2rem",
                "color": "darkgray",
                "border": "none",
            },
            css=[
                {"selector": "tr:hover td", "rule": "color: #91dfd2 !important;"},
                {"selector": "td", "rule": "border: none !important;"},
                {
                    "selector": ".dash-cell.focused",
                    "rule": "background-color: #1e2130 !important;",
                },
                {"selector": "table", "rule": "--accent: #1e2130;"},
                {"selector": "tr", "rule": "background-color: transparent"},
            ],
            data=new_df.to_dict("rows"),
            columns=[{"id": c, "name": c} for c in ["Specs", "Current Setup"]],
        )

@dash_app.callback(Output('h', 'src'), [Input('interval-component', 'n_intervals')])
def change_button_style(Interval):
    if not ((df_bio['PD'][len(df_bio)-1]==0) & (door_idx==1)):
        if ((np.isnan(df[df['CMD']=='60'].index.min())==False) & (np.isnan(df[df['CMD']=='61'].index.min())==True)):
            return dash_app.get_asset_url("bh.png")
        elif ((np.isnan(df[df['CMD']=='60'].index.min())==False) & (df[df['CMD']=='60'].index.min() < df[df['CMD']=='61'].index.min())):
            return [dash_app.get_asset_url("bh.png")]
        elif ((outlier_list[3]>0) | (outlier_list[4]>0) | (outlier_list[5]>0)):
            return dash_app.get_asset_url("bh.png")
        else:
            return [dash_app.get_asset_url("h.png")]
    else:
        return [dash_app.get_asset_url("h.png")]
door_idx = -1
door_time = ''
@dash_app.callback(output = [Output('home', 'src')], inputs = [Input('interval-component', 'n_intervals')])
def change_button_style2(Interval):
    global door_time
    global door_idx
    for t in df[df['Sensor ID']=='4a']['Time']:
        if (t > door_time):
            door_time = t
            door_idx*=-1
    global out    
    if (df_bio['PD'][len(df_bio)-1]==0) & (door_idx==1):
        out = '외출'
        return [dash_app.get_asset_url("go.png")]

    else:
        out = '재실'
        return [dash_app.get_asset_url("home.png")]
        

@dash_app.callback(output = [Output('Temp_pf','children')],inputs = [Input('interval-component', 'n_intervals')])
def change_color1(Interval):
    if outlier_list[0] == -1:
        return [daq.Indicator(
                id="TEMP_indicator", value=True, color="#91dfd2", size=12
            )]
    else:
        return [daq.Indicator(
                id="TEMP_indicator", value=True, color="orange", size=12
            )]

@dash_app.callback(output = [Output('Hum_pf','children')],inputs = [Input('interval-component', 'n_intervals')])
def change_color(Interval):
    if outlier_list[1] == -1:
        return [daq.Indicator(
                id="HUM_indicator", value=True, color="#91dfd2", size=12
            )]
    else:
        return [daq.Indicator(
                id="HUM_indicator", value=True, color="orange", size=12
            )]


@dash_app.callback(output = [Output('PD_pf','children')],inputs = [Input('interval-component', 'n_intervals')])
def change_color3(Interval):
    if outlier_list[2] == -1:
        return [daq.Indicator(
                id="PD_indicator", value=True, color="#91dfd2", size=12
            )]
    else:
        return [daq.Indicator(
                id="PD_indicator", value=True, color="red", size=12
            )]

@dash_app.callback(output = [Output('ACT_pf','children')],inputs = [Input('interval-component', 'n_intervals')])
def change_color4(Interval):
    if outlier_list[3] == -1:
        return [daq.Indicator(
                id="ACT_indicator", value=True, color="#91dfd2", size=12
            )]
    else:
        return [daq.Indicator(
                id="ACT_indicator", value=True, color="red", size=12
            )]

@dash_app.callback(output = [Output('BR_pf','children')],inputs = [Input('interval-component', 'n_intervals')])
def change_color5(Interval):
    if outlier_list[4] == -1:
        return [daq.Indicator(
                id="BR_indicator", value=True, color="#91dfd2", size=12
            )]
    else:
        return [daq.Indicator(
                id="BR_indicator", value=True, color="red", size=12
            )]

@dash_app.callback(output = [Output('HR_pf','children')],inputs = [Input('interval-component', 'n_intervals')])
def change_color6(Interval):
    if outlier_list[5] == -1:
        return [daq.Indicator(
                id="HR_indicator", value=True, color="#91dfd2", size=12
            )]
    else:
        return [daq.Indicator(
                id="HR_indicator", value=True, color="red", size=12
            )]

if __name__ == "__main__":
    application.debug = True
    application.run(host='0.0.0.0')