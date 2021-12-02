
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#data = pd.read_excel('data/nightingale-data.xlsx', sheet_name='Apr1855-Mar1856')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

def hours_of_day_table(df):
    #Processing to set the table with groupings of count of each crime category
    df['Hours'] = df['CrimeDateTime'].str.slice(11,13)
    df['Hours'] = pd.to_numeric(df['Hours'], downcast='integer')
    pd.options.display.float_format = '{:,.0f}'.format

    #print (df[100:1200])
    df_grouped=df.groupby(['Hours', 'Description']).size().reset_index(name='Counts')
    #print (df_grouped)
    
    
    data = df_grouped.pivot_table('Counts', ['Hours'], 'Description').reset_index()
    #print (data)
    data["Hours"]=data["Hours"].astype(int)
    data["Hours"]=data["Hours"].astype(str)
    # data["Test"] = data["Counts"]-250
    print (data)
    return data


def crimes_per_hour_viz(df, categories, static):
    
    data = hours_of_day_table(df)
    
    fig = go.Figure()
    if static:
        #Combining all categories into 3 categories
        categories  = ["Theft, Larceny & Robbery","Assault, Homicide & Rape", "Arson & Shooting"]
        data["Assault, Homicide & Rape"] = data["AGG. ASSAULT"] + data["COMMON ASSAULT"] + data["RAPE"]
        data["Arson & Shooting"] = data["SHOOTING"] + data["ARSON"]
        data["Theft, Larceny & Robbery"] = data["AUTO THEFT"] + data["BURGLARY"] + data["LARCENY"] + data["LARCENY FROM AUTO"] + data["ROBBERY - CARJACKING"] + data["ROBBERY - COMMERCIAL"] + data["ROBBERY - RESIDENCE"] + data["ROBBERY - STREET"]
    
    radius = 0
    #Choose the best 8 categories
    rgb_colors = ["rgb(15,76,129)", "rgb(92,144,144)", "rgb(165,178,181)", "rgb(191,208,202)", "rgb(237,202,190)", "rgb(245,177,156)","rgb(255,162,157)", "rgb(145,88,88)"]
    for i in range(0, len(categories)):
        radius += max(data[categories[i]])
        if i < len(rgb_colors):
            fig.add_trace(go.Barpolar(
                r= data[categories[i]],
                name=categories[i],
                marker_color= rgb_colors[i],
                theta = data['Hours']
            ))

        # fig.add_trace(go.Barpolar(
        #     r= data['Assault, Homicide & Rape'],
        #     name='Assault, Homicide & Rape',
        #     marker_color='rgb(242,188,148)',
        #     theta = data['Hours']
        # ))

        # fig.add_trace(go.Barpolar(
        #     r= data['Arson & Shooting'],
        #     name='Arson & Shooting',
        #     marker_color='rgb(244,175,27)',
        #     theta = data['Hours']
        # ))

        # fig.add_trace(go.Barpolar(
        #     r= data['All other causes Annual rate of mortality per 1000'].tolist(),
        #     name='All other causes Deaths',
        #     marker_color='rgb(239,83,80)',
        #     base = 'stack',
        #     theta = data['Month']
        #  ))


    fig.update_traces(text= data['Hours'].tolist())
    fig.update_layout(
        title='Crimes per Hour of Day',
        title_x=0.5,
        font_size=22,
        legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.75
            ),
        legend_font_size=18,
        template = 'plotly_dark',
        polar = dict(
            radialaxis = dict(range=[0, radius+100]),
            angularaxis = dict( showticklabels=True, showgrid=True, showline=True, rotation=90, direction="clockwise", tickfont_size=16)
        ),
        # angularaxis = dict(
        #     tickfont_size=16,
        #     rotation=90, # start position of angular axis
        #     direction="clockwise"
        #   )
        # ),
    bargroupgap=0
        #showgrid=False
    )

    fig.show(config={'scrollZoom': True})
    #fig.write_image("RoseChart.png", height=500, width=900)


if __name__ == "__main__":
    
	#Importing data- Set path to dataset csv file here
    df = pd.read_csv(r'C:\Users\pauls\Documents\Part1_Crime_data.csv')
    
    #Only Choose upto 8 categories- Should have an error check
    crimes_per_hour_viz(df, ["AGG. ASSAULT","COMMON ASSAULT", "RAPE", "SHOOTING", "ARSON"], False)#, "ARSON", "BURGLARY", "AUTO THEFT", "LARCENY"], True)
    
 
 