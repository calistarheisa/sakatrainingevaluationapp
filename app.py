#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

from scipy.stats import spearmanr

from scipy.stats import ttest_rel, shapiro

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
import base64
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from scipy.stats import shapiro, ttest_rel, wilcoxon

# NLP Model
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fungsi untuk membersihkan teks
def clean_text(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\d+\.', ' ', sentence)
        sentence = re.sub(r'[^\w\s,.\-]', '', sentence)
        sentence = re.sub(r'\n+', ', ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        cleaned_sentences.append(sentence)
    return cleaned_sentences

def add_period_to_sentences(sentences):
    return [sentence if sentence.endswith('.') else sentence + '.' for sentence in sentences]

def sentence_embedding(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def generate_summary(data, title):
    cleaned_data = clean_text(data)
    data_with_periods = add_period_to_sentences(cleaned_data)
    embeddings = sentence_embedding(data_with_periods)
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [data_with_periods[i] for i in np.argsort(scores)[::-1]]
    return ranked_sentences[:5]

def parse_feedback_contents(feedback_contents):
    content_type, content_string = feedback_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_excel(io.BytesIO(decoded))
        df.columns = ["Timestamp", "Judul Pelatihan", "Tempat Pelaksanaan", "Nama Peserta", "Departemen", "Kejelasan Tujuan", "Ketercapaian Tujuan", "Kesesuaian Materi",
                      "Penerapan Materi", "Tambahan Pengetahuan", "Tingkat Kesulitan", "Kemudahan Pemahaman", "Fasilitator Menguasai",
                      "Kompetensi Fasilitator", "Keaktifan Fasilitator", "Penilaian Fasilitator", "Total Penilaian", "Point Utama Materi", 
                      "Kritik dan Saran"]
        df['Tingkat Kesulitan'] = df['Tingkat Kesulitan'].str.extract(r'^(\d+)').astype(int)
        df['Penilaian Fasilitator'] = df['Penilaian Fasilitator'].str.extract(r'^(\d+)').astype(int)
        df['Total Penilaian'] = df['Total Penilaian'].str.extract(r'^(\d+)').astype(int)
        df = df.sort_values(by="Nama Peserta")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def parse_pre_contents(pre_contents):
    content_type, content_string = pre_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_excel(io.BytesIO(decoded))
        df.columns = ["Timestamp", "Nama Peserta", "Nilai Pre-Test"]
        df = df.drop(columns=["Timestamp"])
        df = df.sort_values(by="Nama Peserta")
        return df
    except Exception as e:
        print(f"Error reading nilai file: {e}")
        return None

def parse_post_contents(post_contents):
    content_type, content_string = post_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_excel(io.BytesIO(decoded))
        df.columns = ["Timestamp", "Nama Peserta", "Nilai Post-Test"]
        df = df.drop(columns=["Timestamp"])
        df = df.sort_values(by="Nama Peserta")
        return df
    except Exception as e:
        print(f"Error reading nilai file: {e}")
        return None

external_stylesheets = [
    dbc.themes.BOOTSTRAP,  # Menggunakan Bootstrap
    {
        "href": "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;700&display=swap",
        "rel": "stylesheet",
    },
]

# Dash App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

long_text = """
*Training Evaluation Dashboard* digunakan untuk melakukan visualisasi dan analisis terhadap data evaluasi hasil training. Data evaluasi hasil training harus memenuhi beberapa persayaratan agar dapat digunakan secara maksmimal.

Berikut merupakan persyaratan yang harus dipenuhi:
1. Terdapat 3 data yang perlu diunggah, yaitu: Data Feedback, Data Pre-Test, dan Data Post-Test
2. Setiap data harus terdiri atas kolom-kolom berikut:
- *Data Feedback*: Timestamp, Judul Pelatihan, Tempat Pelaksanaan, Nama Peserta, Departemen, Kejelasan Tujuan, Ketercapaian Tujuan, Kesesuaian Materi, Penerapan Materi, Tambahan Pengetahuan, Tingkat Kesulitan, Kemudahan Pemahaman, "Fasilitator Menguasai, Kompetensi Fasilitator, Keaktifan Fasilitator, Penilaian Fasilitator, Total Penilaian, Point Utama Materi, Kritik dan Saran
- *Data Pre-Test* : Timestamp, Nama Peserta, Nilai Pre-Test
- *Data Post-Test* : Timestamp, Nama Peserta, Nilai Post-Test
Note: Format penamaan setiap kolom tidak harus seperti apa yang telah disebutkan.
3. Setiap data yang diunggah harus dalam format *.xlsx*""" 

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src='/assets/LOGOPGN.jpeg',  # Path to your logo or image
                        style={
                            'maxWidth': '100%'
                        }
                    ),
                    width=3
                ),
                dbc.Col(
                    html.H1(
                        "Training Evaluation Dashboard", 
                        className="text-center",
                        style={"fontFamily": "Poppins", 'fontWeight':'bold','color': '#0056A1','textAlign':'center',
                              'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100%'}
                    ),
                    width=9 
                )
            ],
            style={'alignItems':'center'}
        ),
        dcc.Tabs(
            id='tabs',
            value='tab-1',
            children=[
                dcc.Tab(
                    label='Input Data',
                    style={'padding': '10px', 'font-size': '25px', 'fontFamily': 'Poppins', 'fontWeight': 'bold', 'border': '1px solid #ddd', 'justify-content': 'center', 'align-items': 'center', 'backgroundColor': '#F9F9F9', 'text-align': 'center'},
                    value='tab-1',
                    children=[
                        dbc.Row(
                            [   
                                dbc.Col(
                                    html.Div("Welcome, User!", className="mt-4", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#0056A1', 'textAlign': 'left', 'fontFamily': "Poppins"}),
                                    width=12
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(html.P(dcc.Markdown(long_text), style={'fontSize': 14, 'fontFamily': 'Poppins', 'lineHeight': '2', 'whiteSpace': 'pre-wrap', 'width': '100%', 'color': '#333',
                                                               'border': '1px solid #ddd', 'borderRadius': '8px', 'padding': '15px', 'backgroundColor': '#F9F9F9', 'marginBottom': '20px',
                                                                'textAlign':'justify'})),
                                    width=9
                                ),
                                dbc.Col(
                                    html.Img(
                                        src='/assets/saka1.jpeg',  # Path ke folder assets
                                        style={
                                            'maxWidth': '100%',
                                            'border': '1px solid #ddd',
                                            'borderRadius': '8px',
                                            'padding': '10px',
                                            'backgroundColor': '#F9F9F9'
                                        }
                                    ),
                                    width=3
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Upload(id='upload-feedback', children=html.Button('Upload Data Feedback', style={"fontFamily": "Poppins", "backgroundColor": "#0056A1", "color": "white", 'border': 'none', 'borderRadius': '5px', 'padding': '10px 20px', 'cursor': 'pointer', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1'}), multiple=False), width="auto", className="me-2"),
                                dbc.Col(dcc.Upload(id='upload-pre', children=html.Button('Upload Data Pre-Test', style={"fontFamily": "Poppins", "backgroundColor": "#0056A1", "color": "white", 'border': 'none', 'borderRadius': '5px', 'padding': '10px 20px', 'cursor': 'pointer', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1'}), multiple=False), width="auto", className="me-2"),
                                dbc.Col(dcc.Upload(id='upload-post', children=html.Button('Upload Data Post-Test', style={"fontFamily": "Poppins", "backgroundColor": "#0056A1", "color": "white", 'border': 'none', 'borderRadius': '5px', 'padding': '10px 20px', 'cursor': 'pointer', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1'}), multiple=False), width="auto"),
                            ],
                            justify="start",
                            className="mt-4"
                        ),
                        html.Div(id='upload-status', style={"fontFamily": "Poppins", 'fontSize': 16, 'textAlign': 'left', 'marginTop': '15px'})
                    ]
                ),
                dcc.Tab(
                    label='Output',
                    style={'padding': '10px', 'font-size': '25px', 'fontFamily': 'Poppins', 'fontWeight': 'bold', 'border': '1px solid #ddd', 'justify-content': 'center', 'align-items': 'center', 'backgroundColor': '#F9F9F9', 'text-align': 'center'},
                    value='tab-2',
                    children=[
                         html.Div(
                            id="frame-box",  # Frame box for title and subtitle
                            style={
                                "border": "1px solid #ddd",  # Border color and thickness
                                "padding": "15px",  # Internal spacing
                                "borderRadius": "8px",  # Rounded corners
                                "backgroundColor": "#0056A1",  # Background color
                                "textAlign": "center",  # Center alignment
                                "width": "100%",  # Adjust width to fit content
                                "margin": "auto",  # Center the frame
                                "marginBottom": "10px",  # Space below the frame
                                'marginTop' : '10px'
                            },
                            children=[
                                html.H1(
                                    id="training-title",  # Dynamic title ID
                                    children="Evaluation Dashboard",  # Default title
                                    className="mb-2",
                                    style={"fontFamily": "Poppins", 'color': '#FFFFFF','font-size':'35px','fontWeight':'bold'}
                                ),
                                html.H3(
                                    id="training-subtitle",  # Subtitle ID
                                    children="12/19/2024",  # Default subtitle
                                    className="text-center mb-2",
                                    style={"fontFamily": "Poppins", 'color': '#FFFFFF','font-size':'20px'}
                                ),
                                html.H3(
                                    id="training-location",  # Subtitle ID
                                    children="12/19/2024",  # Default subtitle
                                    className="text-center mb-2",
                                    style={"fontFamily": "Poppins", 'color': '#FFFFFF','font-size':'20px'}
                                ),
                            ]
                        ),
                        dbc.Row(dbc.Col(html.H2("Level 1 : Training Feedback Visualization", className="text-center mt-4", style={"fontFamily": "Poppins", 'color': '#0056A1','font-size':'25px','fontWeight':'bold'}))),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id='Training Score Stacked Bar Chart'), width=12,
                                       style={"display": "flex", "justify-content": "center", "align-items": "center"})
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id='Fasilitator Score Stacked Bar Chart'), width=12,
                                       style={"display": "flex", "justify-content": "center", "align-items": "center"})
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id='Difficulty Level Pie Chart'), width=6,
                                       style={"display": "flex", "justify-content": "center", "align-items": "center"}),
                                dbc.Col(dcc.Graph(id='Facilitator Score Pie Chart'), width=6,
                                       style={"display": "flex", "justify-content": "center", "align-items": "center"})
                            ]
                        ),
                        dbc.Row(
                            dbc.Col(dcc.Graph(id='Total Training Score Bar Chart'), width=12),
                        ),
                        html.Div(id='output-summary', className="mt-4", style={"fontFamily": "Poppins"}),
                        html.Hr(),
                        dbc.Row(dbc.Col(html.H2("Level 2 : Pre-Test dan Post-Test Analysis", className="text-center", style={"fontFamily": "Poppins", 'color': '#0056A1','font-size':'25px','fontWeight':'bold','marginBottom':'30px','marginTop':'10px'}))),
                        dbc.Row(
                            dbc.Col(
                                dash_table.DataTable(
                                    id='pre-post-table',
                                    style_table={'overflowX': 'auto', 'width': '100%', 'margin': '0 auto', 'border': '1px solid #ccc'},
                                    style_cell={
                                        'textAlign': 'center',
                                        'padding': '10px',
                                        'fontFamily': 'Poppins',
                                        'fontSize': '17px',
                                    },
                                    style_cell_conditional=[
                                        {
                                            'if': {'column_id': 'Nama Peserta'},
                                            'textAlign': 'left',
                                        }
                                    ],
                                    style_header={
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                        'fontWeight': 'bold',
                                        'textAlign': 'center',
                                        'fontFamily': 'Poppins',
                                    },
                                    style_data={'backgroundColor': '#ffffff'}
                                )
                            )
                        ),
                        dbc.Row(html.Div("Treshold Point for Post-Test = 70", className="mt-2", style={'fontSize': 14, 'color': '#000000', 'textAlign': 'left', 'fontFamily': "Poppins", 'marginLeft': '20px'})),
                        dbc.Row(
                            [ 
                                dbc.Col(
                                    html.H3("Hasil Analisis", className='mt-4', style={'text-align': 'center', 'fontFamily': 'Poppins', 'fontSize': '25px', 'color': '#0056A1', 'fontWeight': 'bold'}),
                                    width=12)
                            ]),
                        dbc.Row(
                            [    
                                dbc.Col(
                                    dcc.Graph(id='boxplot-pre-post', style={"fontFamily": "Poppins","width":"100%","height":"600px"}), 
                                    width=6,
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                    },
                                ),
                                dbc.Col(
                                    dbc.Row(
                                        [
                                            # Statistical Results Box
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.H4("Apakah ada peningkatan nilai setelah pelatihan?",
                                                                className="card-title", style={'text-align': 'left', 'color': '#0056A1', 'fontsize': '15px'}),
                                                        html.Div(id='statistical-test-results', className="mt-3", style={"fontFamily": "Poppins"})
                                                    ]),
                                                ),
                                                width=12,
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                },
                                            ),

                                            # Correlation Results Box
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.H4("Apakah ada pengaruh antara nilai post-test dengan pilihan tingkat kesulitan materi oleh peserta?", 
                                                                className="card-title", style={'text-align': 'left', 'color': '#0056A1', 'fontsize': '15px'}),
                                                        html.Div(id='correlation-results', className="mt-3", style={"fontFamily": "Poppins"})
                                                    ]),
                                                ),
                                                width=12,
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                },
                                            ),
                                        ]
                                    ),
                                    width=6,
                                    style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    },
                                ),
                            ],
                            style={
                                "alignItems": "center",
                                "justifyContent": "center",
                                "padding": "20px",
                            },
                        ),
                    ]
                ),
            ]
        ),
    ],
    fluid=True
)



@app.callback(
    [Output('Training Score Stacked Bar Chart', 'figure'),
     Output('Fasilitator Score Stacked Bar Chart', 'figure'),
     Output('Difficulty Level Pie Chart', 'figure'),
     Output('Facilitator Score Pie Chart', 'figure'),
     Output('Total Training Score Bar Chart', 'figure'),
     Output('output-summary', 'children'),
     Output('pre-post-table', 'data'),
     Output('pre-post-table', 'columns'),
     Output('boxplot-pre-post', 'figure'),
     Output('statistical-test-results', 'children'),
     Output('correlation-results', 'children'),
     Output('upload-status','children'),
     Output('training-title', 'children'),
     Output('training-subtitle','children'),
     Output('training-location','children')
    ],
    [Input('upload-feedback', 'contents'),
     Input('upload-pre','contents'),
     Input('upload-post','contents')
    ],
    [State('upload-feedback', 'filename'),
     State('upload-pre', 'filename'),
     State('upload-post', 'filename')
    ]
)

def update_graph(feedback_contents, pre_contents, post_contents, feedback_filename, pre_filename, post_filename):        
    # Default
    empty_fig = go.Figure().update_layout(
        title="Data Belum Diunggah", xaxis_title="", yaxis_title="",title_font=dict(family="Poppins", size=20)
    )
    empty_summary = html.Div("Data belum diunggah.",style={'color':'#3A8FB7','fontFamily':'Poppins'})
    empty_table_data = []
    empty_table_columns = [{"name": " ", "id": " "}] 
    empty_statistical_results = html.Div("Hasil uji statistik akan ditampilkan di sini.",style={'fontFamily':'Poppins'})
    empty_correlation_results = html.Div("Hasil uji korelasi akan ditampilkan di sini.",style={'fontFamily':'Poppins'})
    upload_status = html.Div("Data belum diunggah.",style={'color':'#0056A1','fontFamily':'Poppins'})
    default_title = html.Div('Evaluation Dashboard',style={'fontFamily':'Poppins'})
    default_subtitle = html.Div('Tanggal Tidak Ditemukan',style={'fontFamily':'Poppins'})
    default_location = html.Div('Tempat Pelaksanaan Tidak Ditemukan',style={'fontFamily':'Poppins'})

    if feedback_contents is None or pre_contents is None or post_contents is None:
        return (
            empty_fig, 
            empty_fig, 
            empty_fig,
            empty_fig,
            empty_fig,
            empty_summary, 
            empty_table_data, 
            empty_table_columns, 
            empty_fig, 
            empty_statistical_results, 
            empty_correlation_results,
            upload_status,
            default_title,
            default_subtitle,
            default_location
        )

    if not feedback_filename.endswith('.xlsx') or not pre_filename.endswith('.xlsx') or not post_filename.endswith('.xlsx'):
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            html.Div("Silakan unggah file Excel yang valid.", style={"color": "red"}),
            empty_table_data,
            empty_table_columns,
            empty_fig,
            empty_statistical_results,
            empty_correlation_results,
            upload_status,
            default_title,
            default_subtitle,
            default_location
        )

    # Parse file content and handle errors
    try:
        feedback_data = parse_feedback_contents(feedback_contents)
        pre_data = parse_pre_contents(pre_contents)
        post_data = parse_post_contents(post_contents)
        if feedback_data is None or pre_data is None or post_data is None:
            return (
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                html.Div("Terjadi kesalahan saat membaca file.", style={"color": "red"}),
                empty_table_data,
                empty_table_columns,
                empty_fig,
                empty_statistical_results,
                empty_correlation_results,
                upload_status,
                default_title,
                default_subtitle,
                default_location
            )
        
        upload_status = html.Div("Data berhasil diproses. Silakan ke tab Output untuk melihat hasil analisis.", style={"color": "green","fontFamily":"Poppins"})
     
        # Extract title from 'Judul Pelatihan' column (if exists)
        if 'Judul Pelatihan' in feedback_data.columns:
            extracted_title = feedback_data['Judul Pelatihan'].iloc[0]  # Ambil nilai pertama dari kolom 'Judul Pelatihan'
        else:
            extracted_title = default_title
        
        # Extract date from 'Timestamp' column (if exists)
        if 'Timestamp' in feedback_data.columns:
            # Convert timestamp to datetime and extract the date
            extracted_date = pd.to_datetime(feedback_data['Timestamp'].iloc[0]).strftime('%d %B %Y')
        else:
            extracted_date = "Tanggal Tidak Ditemukan"  # Default if no timestamp column
        
        # Extract location from 'Tempat Pelaksanaan' column (if exists)
        if 'Tempat Pelaksanaan' in feedback_data.columns:
            extracted_location = feedback_data['Tempat Pelaksanaan'].iloc[0]  # Ambil nilai pertama dari kolom 'Tempat Pelaksanaan'
        else:
            extracted_location = default_location  # Jika kolom tidak ada, gunakan tempat default
            
        pre_data.drop(columns=['Nama Peserta'], axis=1, inplace=True)
        post_data.drop(columns=['Nama Peserta'], axis=1, inplace=True)

        combined_data = pd.concat([feedback_data, pre_data, post_data], axis=1)

        categories = ['Ya', 'Tidak']
        # Training Score Chart
        training = combined_data[["Kejelasan Tujuan", "Ketercapaian Tujuan", "Kesesuaian Materi", "Penerapan Materi", "Tambahan Pengetahuan"]]
        training_counts = pd.DataFrame({
            col: training[col].value_counts(normalize=True) * 100
            for col in training.columns
        }).fillna(0)
        category_colors = {
                'Ya': "#0056A1", 
                'Tidak': '#3A8FB7', 
        }
        training_fig = go.Figure()
        for cat in categories:
            if cat == 'Ya':
                training_fig.add_trace(go.Bar(
                    x=training_counts.columns,
                    y=training_counts.loc[cat] if cat in training_counts.index else [0]*len(training_counts.columns),
                    name=cat,
                    marker_color=category_colors.get(cat, "#ffffff"),
                    text=training_counts.loc[cat].round(1).astype(str) + '%',
                    textposition='inside',  
                    texttemplate='%{text}',
                    offsetgroup = cat,
                    width = 0.6
                ))
            else:
                training_fig.add_trace(go.Bar(
                    x=training_counts.columns,
                    y=training_counts.loc[cat] if cat in training_counts.index else [0]*len(training_counts.columns),
                    name=cat,  
                    marker_color=category_colors.get(cat, "#ffffff"),
                    offsetgroup = cat,
                    width = 0.6
                ))      
            
        training_fig.update_layout(
            barmode='stack',
            title='Training Score Stacked Bar Chart',
            title_font=dict(family="Poppins", size=20,weight='bold'),
            font=dict(family="Poppins"),
            xaxis_title='Questions',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, 100],showgrid=False),
            xaxis=dict(
                    showgrid=False, 
                    tickangle=0,  # Label horizontal
                    tickfont=dict(size=12),  # Memberikan jarak lebih besar antar label
                    automargin=True,
                    ticktext=[
                        "Kejelasan<br>Tujuan",
                        "Ketercapaian<br>Tujuan",
                        "Kesesuaian<br>Materi",
                        "Penerapan<br>Materi",
                        "Tambahan<br>Pengetahuan"
                            ],
                    tickvals = list(range(len(training_counts.columns)))
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            bargap = 0.4
        )

        # Fasilitator Score Chart
        fasil = combined_data[['Kemudahan Pemahaman', 'Fasilitator Menguasai', 'Kompetensi Fasilitator', 'Keaktifan Fasilitator']]
        fasil_counts = pd.DataFrame({
            col: fasil[col].value_counts(normalize=True) * 100
            for col in fasil.columns
        }).fillna(0)
        category_colors = {
                'Ya': "#0056A1",  # Biru
                'Tidak': '#3A8FB7',  # Biru Muda
        }
        fasil_fig = go.Figure()
        for cat in categories:
            if cat == "Ya":
                fasil_fig.add_trace(go.Bar(
                    x=fasil_counts.columns,
                    y=fasil_counts.loc[cat] if cat in fasil_counts.index else [0]*len(fasil_counts.columns),
                    name=cat,
                    marker_color=category_colors.get(cat, "#ffffff"),
                    text=fasil_counts.loc[cat].round(1).astype(str) + '%',
                    textposition='inside',  
                    texttemplate='%{text}',
                    offsetgroup =cat,
                    width = 0.6
                ))
            else:
                fasil_fig.add_trace(go.Bar(
                    x=fasil_counts.columns,
                    y=fasil_counts.loc[cat] if cat in fasil_counts.index else [0]*len(fasil_counts.columns),
                    name=cat,
                    marker_color=category_colors.get(cat, "#8AA0B2"),
                    offsetgroup=cat,
                    width = 0.6
                ))
                
        fasil_fig.update_layout(
            barmode='stack',
            title='Fasilitator Score Stacked Bar Chart',
            title_font=dict(family="Poppins", size=20, weight='bold'),
            font=dict(family="Poppins"),
            xaxis_title='Questions',
            yaxis_title='Percentage (%)',
            yaxis=dict(range=[0, 100],showgrid=False),
            xaxis=dict(
                    showgrid=False, 
                    tickangle=0,  # Label horizontal
                    tickfont=dict(size=12),  # Memberikan jarak lebih besar antar label
                    automargin=True,
                    ticktext=[
                        "Kemudahan<br>Pemahaman",
                        "Fasilitator<br>Menguasai",
                        "Kompetensi<br>Fasilitator",
                        "Keaktifan<br>Fasilitator"
                            ],
                    tickvals = list(range(len(training_counts.columns)))
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            bargap = 0.4
        )

        # Difficulty Level Pie Chart
        diff_level = combined_data['Tingkat Kesulitan'].dropna()
        difficulty_categories = [5, 4, 3, 2, 1]
        diff_frequencies = diff_level.value_counts().reindex(difficulty_categories, fill_value=0)
        custom_labels_diff = ['Sangat Sulit', 'Sulit', 'Biasa Saja', 'Mudah', 'Sangat Mudah']

        pie_chart = go.Figure(data=[go.Pie(
            labels=custom_labels_diff,
            values=diff_frequencies,
            textinfo='percent',
            insidetextorientation='auto',
            hole=0.3,
            marker=dict(colors=["#0056A1", '#3A8FB7', '#66A1D9', '#ADD8E6',"#8AA0B2"]),
            showlegend=True,
            sort=False
        )])

        # Add title to Difficulty Level Pie Chart
        pie_chart.update_layout(
            title="Distribusi Level Kesulitan",
            title_font=dict(family="Poppins", size=20, weight='bold'),
            font=dict(family="Poppins"),
            title_x=0.5,
            yaxis=dict(visible=False),
            xaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            width=450,
            showlegend=True
        )

        # Facilitator Score Pie Chart
        fasil_score = combined_data['Penilaian Fasilitator'].dropna()
        score_frequencies = fasil_score.value_counts().reindex(difficulty_categories, fill_value=0)
        custom_labels_score = ['Sangat Tinggi', 'Tinggi', 'Sedang', 'Rendah', 'Sangat Rendah']        

        score_pie_chart = go.Figure(data=[go.Pie(
            labels=custom_labels_score,
            values=score_frequencies,
            textinfo='percent',
            insidetextorientation='auto',
            hole=0.3,
            marker=dict(colors=["#0056A1", '#3A8FB7', '#66A1D9', '#ADD8E6',"#8AA0B2"]),
            showlegend=True,
            sort=False
        )])

        # Add title to Facilitator Score Pie Chart
        score_pie_chart.update_layout(
            title="Distribusi Nilai Fasilitator",
            title_font=dict(family="Poppins", size=20,weight='bold'),
            font=dict(family="Poppins"),
            title_x=0.5,
            yaxis=dict(visible=False),
            xaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            width=450,
            showlegend=True
        )

        # Total Score Bar Chart
        total_score = combined_data['Total Penilaian'].dropna()
        total_categories = [5, 4, 3, 2, 1]
        total_frequencies = total_score.value_counts().reindex(total_categories, fill_value=0)
        custom_labels_total = ['Exelence', 'Baik', 'Sedang', 'Rendah', 'Sangat Rendah']
        # Buat bar chart horizontal
        custom_labels_total = custom_labels_total[::-1]
        total_frequencies = total_frequencies[::-1]
        total_score_fig = go.Figure(data=[go.Bar(
            y=custom_labels_total,  
            x=total_frequencies,    
            orientation='h',       
            text=total_frequencies, 
            textposition='auto',
            marker=dict(color='#0056A1')
        )])  
        total_score_fig.update_layout(
            title="Distribusi Nilai Training Score",
            title_font=dict(family="Poppins", size=20,weight='bold'),
            font=dict(family="Poppins"),
            title_x=0.5,
            xaxis_title="Frequency",
            yaxis_title="Training Score",
            showlegend=False,
            yaxis=dict(showgrid=False),
            xaxis=dict(showgrid=False),
            plot_bgcolor='white',
            paper_bgcolor='white'   
        )

        # Tambahkan placeholder untuk kategori dengan frekuensi 0
        for i, freq in enumerate(total_frequencies):
            if freq == 0:
                total_score_fig.add_trace(go.Scatter(
                    x=[0], y=[custom_labels_total[i]],
                    mode='markers',
                    marker=dict(color='lightgrey', size=10, symbol='square'),
                    name=f"{custom_labels_total[i]} (0)",
                    showlegend=True
                ))

        # Generate Summary
        material_summary = generate_summary(combined_data["Point Utama Materi"].dropna(), "Material")
        suggestion_summary = generate_summary(combined_data["Kritik dan Saran"].dropna(), "Suggestion")

        summary_component = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Kesimpulan Material", className="card-title", style={"fontFamily":"Poppins",'color': '#0056A1','fontWeight':'bold'}),
                                html.Ul([html.Li(sentence) for sentence in material_summary]),
                            ]
                        )
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Kesimpulan Saran", className="card-title",style={"fontFamily":"Poppins",'color': '#0056A1','fontWeight':'bold'}),
                                html.Ul([html.Li(sentence) for sentence in suggestion_summary]),
                            ]
                        )
                    ),
                    width=6
                ),
            ],
            className="mt-4"
        )

        # Pre-test and Post-test Table
        sorted_data = combined_data[["Nama Peserta", "Nilai Pre-Test", "Nilai Post-Test"]].sort_values(by="Nama Peserta")
        sorted_data['No.'] = range(1, len(sorted_data) + 1)
        sorted_data['No.'] = sorted_data['No.'].astype(str) + '.'
        sorted_data['Gain'] = sorted_data['Nilai Post-Test'] - sorted_data['Nilai Pre-Test']
        sorted_data = sorted_data.rename(columns={'Gain': 'Gain(Δ)'})
        sorted_data = sorted_data[['No.', 'Nama Peserta', 'Nilai Pre-Test', 'Nilai Post-Test','Gain(Δ)']]
        table_data = sorted_data.to_dict('records')
        table_columns = [{"name": col, "id": col} for col in sorted_data.columns]


        # Boxplot for Pre-test and Post-test
        data_nilai = combined_data[["Nilai Pre-Test", "Nilai Post-Test"]].melt(var_name="Kategori", value_name="Nilai")
        data_nilai["Kategori"] = data_nilai["Kategori"].replace({"Nilai Pre-Test": "Pre-Test", "Nilai Post-Test": "Post-Test"})
        boxplot_fig = px.box(
            data_nilai, 
            x="Kategori", 
            y="Nilai", 
            color="Kategori",
            color_discrete_sequence=["#0056A1"],
            title="Box Plot",
            height=800,
        )
        boxplot_fig.update_layout(xaxis_title="Kategori", yaxis_title="Nilai", title_font=dict(size=24, color='#0056A1',family='Poppins',weight='bold'),font=dict(family="Poppins", size=15), showlegend=False)

        # Statistical Test
        stat_pre, p_pre = shapiro(combined_data["Nilai Pre-Test"])
        stat_post, p_post = shapiro(combined_data["Nilai Post-Test"])

        if p_pre > 0.05 and p_post > 0.05:
            t_stat, p_value = ttest_rel(combined_data["Nilai Post-Test"], combined_data["Nilai Pre-Test"])
            p_value_one_tailed = p_value / 2 *100
            
            if p_value_one_tailed < 0.05:
                statistical_results = f"Pengujian secara statistik menunjukkan hasil yang signifikan bahwa rata-rata post-test lebih tinggi dibandingkan dengan nilai pre-test karena memiliki tingkat kesalahan sebesar {p_value_one_tailed:.2f}% yang lebih kecil dari 5%."
            else:
                statistical_results = f"Pengujian secara statistik tidak menunjukkan hasil yang signfikan bahwa rata-rata nilai post-test lebih tinggi dibandingkan dengan nilai pre-test karena memiliki tingkat keselahan dengan tingkat kesalahan sebesar {p_value_one_tailed:.2f}% yang lebih besar dari 5%."
        else:
            w_stat, p_wilcoxon = wilcoxon(combined_data["Nilai Post-Test"], combined_data["Nilai Pre-Test"], alternative='greater')
                        
            if p_wilcoxon < 0.05:
                statistical_results = f"Pengujian secara statistik menunjukkan hasil yang signifikan bahwa terdapat peningkatan nilai setelah pelatihan karena memiliki tingkat kesalahan pengujian sebesar {p_wilcoxon*100:.2f}% yang lebih kecil dari 5%."
            else:
                statistical_results = f"Pengujian secara statistik tidak menunjukkan hasil yang signfikan bahwa terdapat peningkatan nilai setelah pelatihan karena memiliki tingkat kesalahan pengujian sebesar sebesar {p_wilcoxon*100:.2f}% yang lebih besar dari 5%."
                

        # Correlation
        stat, p_value = spearmanr(combined_data["Tingkat Kesulitan"], combined_data["Nilai Post-Test"])

        if p_value < 0.05:
            if stat < 0:
                correlation_results = f"Secara statistik, diperoleh kesimpulan bahwa terdapat pengaruh antara pilihan tingkat kesulitan materi oleh peserta dengan nilai post-test yang didapatkan. Di mana nilai korelasi {stat:.2f} yang bertanda negatif menunjukan bahwa peserta yang semakin menganggap materi pelatihan sulit mendapatkan nilai post-test yang semakin kecil, atau sebaliknya."
            else:
                correlation_results = f"Secara statistik, diperoleh kesimpulan bahwa terdapat pengaruh antara pilihan tingkat kesulitan materi oleh peserta dengan nilai post-test yang didapatkan. Di mana nilai korelasi {stat:.2f} yang bertanda positif menunjukan bahwa peserta yang semakin menganggap materi pelatihan sulit mendapatkan nilai post-test yang semakin besar, atau sebaliknya."
        else:
            correlation_results = f"Secara statistik, diperoleh kesimpulan bahwa tidak terdapat pengaruh antara pilihan tingkat kesulitan materi oleh peserta dengan nilai post-test yang didapatkan."

        return (
            training_fig, 
            fasil_fig, 
            pie_chart, 
            score_pie_chart, 
            total_score_fig, 
            summary_component, 
            table_data, 
            table_columns, 
            boxplot_fig, 
            html.Div(html.P(statistical_results)), 
            html.Div(html.P(correlation_results)),
            upload_status,
            extracted_title,
            extracted_date,
            extracted_location
        )
        
    except Exception as e:
            return (
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            html.Div(f"Terjadi kesalahan: {str(e)}", style={"color": "red"}),
            empty_table_data,
            empty_table_columns,
            empty_fig,
            empty_statistical_results,
            empty_correlation_results,
            html.Div(f"Terjadi kesalahan: {str(e)}", style={"color": "red","fontFamily":"Poppins"}),
            default_title,
            default_subtitle,
            default_location
        )


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)

