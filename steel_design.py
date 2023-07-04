import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
pn.extension('plotly')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a DataFrame with sample data
df_all = pd.read_excel('Steel_Table_Python.xlsx')
df_all.set_index('Section', inplace=True)

E_default = 2000000
Fy_default = 2450
c = 1 # eq F2-8a
c1 = 0.22 # FOR LOCAL BUCKLING
c2 = 1.49 # FOR LOCAL BUCKLING

def section_classification_all(E = E_default, Fy = Fy_default, Cb = 1.0, decimal = 2):
    
    global data
    
    data = df_all
    
    # width-to-thickness ratio
    data['lambda_f'] = data['0.5bf/tf']
    data['lambda_w'] = data['h/tw']

    # classification of compression element
    data['c_lambda_rf'] = round(0.56 * np.sqrt(E / Fy), decimal)
    data['c_lambda_rw'] = round(1.49 * np.sqrt(E / Fy), decimal)

    data['Compression: Flange Classification'] = np.where(data['lambda_f'] >= data['c_lambda_rf'], 'Slender', 'NonSlender')
    data['Compression: Web Classification'] = np.where(data['lambda_w'] >= data['c_lambda_rw'], 'Slender', 'NonSlender')

    # classification of flexural element
    data['f_lambda_pf'] = round(0.38 * np.sqrt(E / Fy), decimal)
    data['f_lambda_rf'] = round(1.00 * np.sqrt(E / Fy), decimal)
    data['f_lambda_pw'] = round(3.76 * np.sqrt(E / Fy), decimal)
    data['f_lambda_rw'] = round(5.70 * np.sqrt(E / Fy), decimal)

    data['Flexural: Flange Classification'] = np.where(data['lambda_f'] >= data['f_lambda_rf'], 'Slender',
                                                     np.where((data['f_lambda_rf'] > data['lambda_f']) &
                                                              (data['lambda_f'] >= data['f_lambda_pf']),
                                                              'NonCompact', 'Compact'))
    data['Flexural: Web Classification'] = np.where(data['lambda_w'] >= data['f_lambda_rw'], 'Slender',
                                                  np.where((data['f_lambda_rw'] > data['lambda_w']) &
                                                           (data['lambda_w'] >= data['f_lambda_pw']),
                                                           'NonCompact', 'Compact'))
    
    ry = data['ry [cm]']
    
    data['Lp [m]'] = round(1.76*ry*np.sqrt(E/Fy) / 100, decimal)

    J = data['J [cm4]']
    Sx = data['Sx [cm3]']
    h0 = data['h0 [mm]']
    rts = data['rts [cm]']
    
    aa = E/(0.7*Fy)
    bb = J*c/(Sx*h0/10)
    data['Lr [m]'] = round(1.95*rts*aa*np.sqrt(bb+np.sqrt(bb**2+6.76/(aa**2))) / 100, decimal)
    
    Myx = Fy*data['Sx [cm3]'] /100/1000
    Mpx = Fy*data['Zx [cm3]'] /100/1000
    
    data['My x [ton-m]'] = round(Myx, decimal) # ton-m
    data['Mp x [ton-m]'] = round(Mpx, decimal) # ton-m
    data['Mr x [ton-m]'] =  np.minimum(round(Cb*0.7*Myx, decimal), round(Mpx, decimal))
    
    Myy = Fy*data['Sy [cm3]'] /100/1000
    Mpy = Fy*data['Zy [cm3]'] /100/1000
       
    data['My y [ton-m]'] = round(Myy, decimal) # ton-m
    data['Mp y [ton-m]'] = round(Mpy, decimal) # ton-m
    
    lambda_f = data['lambda_f']
    f_lambda_pf = data['f_lambda_pf']
    f_lambda_rf = data['f_lambda_rf']
    
    if data['Flexural: Flange Classification'][0] == 'Compact':
        data['Mn_minor [ton-m]'] =  np.minimum(round(1.6*Myy, decimal), round(Mpy, decimal))
    elif data['Flexural: Flange Classification'][0] == 'NonCompact':
        data['Mn_minor [ton-m]'] = round(Mpy - (Mpy-0.7*Myy)*((lambda_f-f_lambda_pf)/(f_lambda_rf-f_lambda_pf)), decimal)
    elif data['Flexural: Flange Classification'][0] == 'Slender':
        data['Mn_minor [ton-m]'] = round((0.69*E/((0.5*data['bf [mm]']/data['tf [mm]'])**2))*data['Sy [cm3]'] /100/1000, decimal)
    
    data['2.24 sqrt(E/Fy)'] = round(2.24*np.sqrt(E/Fy), decimal)
    data['Vn [ton]'] = round(0.6*Fy*data['d [mm]']*data['tw [mm]'] /100/1000, decimal)
    
    

section_default = df_all.index[0]

def section_classification(E = E_default, Fy = Fy_default, section = section_default, Cb = 1.0, decimal = 2):
    
    global df
    
    df = pd.DataFrame(df_all.loc[section]).T

    # width-to-thickness ratio
    df['lambda_f'] = df['0.5bf/tf']
    df['lambda_w'] = df['h/tw']

    # classification of compression element
    df['c_lambda_rf'] = round(0.56 * np.sqrt(E / Fy), decimal)
    df['c_lambda_rw'] = round(1.49 * np.sqrt(E / Fy), decimal)
    
    df['Compression: Flange Classification'] = np.where(df['lambda_f'] >= df['c_lambda_rf'], 'Slender', 'NonSlender')
    df['Compression: Web Classification'] = np.where(df['lambda_w'] >= df['c_lambda_rw'], 'Slender', 'NonSlender')

    # classification of flexural element
    df['f_lambda_pf'] = round(0.38 * np.sqrt(E / Fy_default), decimal)
    df['f_lambda_rf'] = round(1.00 * np.sqrt(E / Fy_default), decimal)
    df['f_lambda_pw'] = round(3.76 * np.sqrt(E / Fy_default), decimal)
    df['f_lambda_rw'] = round(5.70 * np.sqrt(E / Fy_default), decimal)

    df['Flexural: Flange Classification'] = np.where(df['lambda_f'] >= df['f_lambda_rf'], 'Slender',
                                                     np.where((df['f_lambda_rf'] > df['lambda_f']) &
                                                              (df['lambda_f'] >= df['f_lambda_pf']),
                                                              'NonCompact', 'Compact'))
    df['Flexural: Web Classification'] = np.where(df['lambda_w'] >= df['f_lambda_rw'], 'Slender',
                                                  np.where((df['f_lambda_rw'] > df['lambda_w']) &
                                                           (df['lambda_w'] >= df['f_lambda_pw']),
                                                           'NonCompact', 'Compact'))
    
    ry = df['ry [cm]'][0]
    
    df['Lp [m]'] = round(1.76*ry*np.sqrt(E/Fy) / 100, decimal)

    J = df['J [cm4]'][0]
    Sx = df['Sx [cm3]'][0]
    Zx = df['Zx [cm3]'][0]
    h0 = df['h0 [mm]'][0]
    rts = df['rts [cm]'][0]

    aa = E/(0.7*Fy)
    bb = J*c/(Sx*h0/10)
    df['Lr [m]'] = round(1.95*rts*aa*np.sqrt(bb+np.sqrt(bb**2+6.76/(aa**2))) / 100, decimal)
    
    Myx = Fy*Sx /100/1000 # ton-m
    Mpx = Fy*Zx /100/1000 # ton-m
    
    df['My x [ton-m]'] = round(Myx, decimal)
    df['Mp x [ton-m]'] = round(Mpx, decimal)
    df['Mr x [ton-m]'] = round(min(Cb*0.7*Myx, Mpx), decimal)
    
    Sy = df['Sy [cm3]'][0]
    Zy = df['Zy [cm3]'][0]
    
    Myy = Fy*Sy /100/1000 # ton-m
    Mpy = Fy*Zy /100/1000 # ton-m
    
    df['My y [ton-m]'] = round(Myy, decimal)
    df['Mp y [ton-m]'] = round(Mpy, decimal)
    
    lambda_f = df['lambda_f'][0]
    f_lambda_pf = df['f_lambda_pf'][0]
    f_lambda_rf = df['f_lambda_rf'][0]
    bf = df['bf [mm]'][0]
    tf = df['tf [mm]'][0]
    
    if df['Flexural: Flange Classification'][0] == 'Compact':
        df['Mn_minor [ton-m]'] = round(min(Mpy, 1.6*Myy), decimal)
    elif df['Flexural: Flange Classification'][0] == 'NonCompact':
        df['Mn_minor [ton-m]'] = round(Mpy - (Mpy-0.7*Myy)*((lambda_f-f_lambda_pf)/(f_lambda_rf-f_lambda_pf)), decimal)
    elif df['Flexural: Flange Classification'][0] == 'Slender':
        df['Mn_minor [ton-m]'] = round((0.69*E/((0.5*bf/tf)**2))*Sy /100/1000, decimal)
        
    df['2.24 sqrt(E/Fy)'] = round(2.24*np.sqrt(E/Fy), decimal)
    df['Vn [ton]'] = round(0.6*Fy*df['d [mm]'][0]*df['tw [mm]'][0] /100/1000, decimal)
        


def compression_cal(E = E_default, Fy = Fy_default, braced_minor = 1.0, braced_major = 1.0, braced_torsion = 1.0):
    
    comp_data = ['Lc [m]',
                 'Fe_FB [ksc]', 'Fcr_FB [ksc]', 'Pn_FB [ton]',
                 'Fe_FTB [ksc]', 'Fcr_FTB [ksc]', 'Pn_FTB [ton]',
                 'Fe_min [ksc]', 'Fcr_min [ksc]', 'Pn_min [ton]',
                 'A_LB [cm2]', 'Pn_min_LB [ton]']

    slenderness_TR = round(4.71*np.sqrt(E/Fy), 2)

    step = 0.5
    slenderness = np.arange(step, slenderness_TR, step)
    slenderness = np.append(slenderness, [slenderness_TR])
    slenderness = np.append(slenderness, np.arange(round(slenderness_TR,0), 200+step, step))

    df_C = pd.DataFrame(index = slenderness, columns = comp_data).reset_index()
    df_C.columns.values[0] = 'Slenderness'

    def Fcr_cal(out_data, in_data):
        ROW = Fy/df_C.loc[:,in_data] <= 2.25
        df_C.loc[ROW,out_data] = 0.658**(Fy/df_C.loc[ROW,in_data])*Fy
        df_C.loc[~ROW,out_data] = 0.877*df_C.loc[~ROW,in_data]

    # SECTION DATA
    A = df['A [cm2]'][0]
    d = df['d [mm]'][0]
    rx = df['rx [cm]'][0]
    ry = df['ry [cm]'][0]
    Cw = df['Cw [cm6]'][0]
    J = df['J [cm4]'][0]
    Ix = df['Ix [cm4]'][0]
    Iy = df['Iy [cm4]'][0]
    lambda_w = df['lambda_w'][0]
    lambda_f = df['lambda_f'][0]
    tw = df['tw [mm]'][0]
    tf = df['tf [mm]'][0]
    bf = df['bf [mm]'][0]

    web_classification = df['Compression: Web Classification'][0]
    c_lambda_rw = df['c_lambda_rw'][0]
    flange_classification = df['Compression: Flange Classification'][0]
    c_lambda_rf = df['c_lambda_rf'][0]

    # EULER BUCKLING STRESS
    slenderness_minor = braced_minor / ry
    slenderness_major = braced_major / rx
    
    if slenderness_major > slenderness_minor:
        message = 'Flexural Buckling around MAJOR axis'
        braced = braced_major
        radius_of_gyration = rx
    else:
        message = 'Flexural Buckling around MINOR axis'
        braced = braced_minor
        radius_of_gyration = ry

    df_C['Lc [m]'] = slenderness * radius_of_gyration / braced / 100
    Lcz = df_C['Lc [m]'] * braced_torsion * 100

    # E3. FLEXURAL BUCKLING OF MEMBERS WITHOUT SLENDER ELEMENTS 
    df_C['Fe_FB [ksc]'] = (np.pi**2)*E/(slenderness**2)
    Fcr_cal('Fcr_FB [ksc]', 'Fe_FB [ksc]')
    df_C['Pn_FB [ton]'] = df_C['Fcr_FB [ksc]'] * A / 1000

    # E4: TORSIONAL AND FLEXURAL-TORSIONAL BUCKLING OF SINGLE ANGLES AND MEMBERS WITHOUT SLENDER ELEMENTS
    G = E/(2*(1+0.3))
    df_C['Fe_FTB [ksc]'] = ((np.pi**2*E*Cw) / (Lcz.pow(2)) + G*J) * (1/(Ix+Iy))
    Fcr_cal('Fcr_FTB [ksc]', 'Fe_FTB [ksc]')
    df_C['Pn_FTB [ton]'] = df_C['Fcr_FTB [ksc]'] * A / 1000

    # MINIMUM Fe FROM E3 and E4
    df_C['Fe_min [ksc]'] = df_C[['Fe_FB [ksc]','Fe_FTB [ksc]']].min(axis=1)

    # E3. FLEXURAL BUCKLING OF MEMBERS WITHOUT SLENDER ELEMENTS 
    df_C['Fcr_min [ksc]'] = df_C[['Fcr_FB [ksc]','Fcr_FTB [ksc]']].min(axis=1)

    # E7: MEMBERS WITH SLENDER ELEMENTS
    df_C['A_LB [cm2]'] = A
    if web_classification == 'Slender':
        ROW = lambda_w > c_lambda_rw*(Fy/df_C['Fcr_min [ksc]']).pow(0.5)
        Fel = Fy*(c2*c_lambda_rw/lambda_w)**0.5
        FF = (Fel/df_C.loc[ROW,'Fcr_min [ksc]']).pow(0.5)
        de = d*(1-c1*FF)*FF

        df_C.loc[ROW,'A_LB [cm2]'] = df_C.loc[ROW,'A_LB [cm2]'] - (d - de) * tw / 100

    if flange_classification == 'Slender':
        ROW = lambda_f > c_lambda_rf*(Fy/df_C['Fcr_min [ksc]']).pow(0.5)
        Fel = Fy*(c2*c_lambda_rf/lambda_f)**0.5
        FF = (Fel/df_C.loc[ROW,'Fcr_min [ksc]']).pow(0.5)
        bfe = bf*(1-c1*FF)*FF

        df_C.loc[ROW,'A_LB [cm2]'] = df_C.loc[ROW,'A_LB [cm2]'] - (bf - bfe) * tf / 100


    # NOMINAL AXIAL FORCE
    df_C['Pn_min [ton]'] = df_C['Fcr_min [ksc]'] * A /1000
    df_C['Pn_min_LB [ton]'] = df_C['Fcr_min [ksc]'] * df_C['A_LB [cm2]'] /1000
    
    df_C = df_C.round(2)
    
    return df_C, slenderness_TR, message
    

def compression_plot(df_C, slenderness_TR, message):
    
    if 'MAJOR' in message:
        xaxis_title = 'Slenderness MAX, Lc/rx'
        xaxis2_title = 'Effective Length, Lc = Slenderness MAX * rx [m]'
    else:
        xaxis_title = 'Slenderness MAX, Lc/ry'
        xaxis2_title = 'Effective Length, Lc = Slenderness MAX * ry [m]'
    
    x_data = df_C['Slenderness']
    x_limit = slenderness_TR
    xx = df_C.loc[df_C['Slenderness'] == slenderness_TR, 'Lc [m]'].values[0]
    y_limit = df_C.loc[df_C['Slenderness'] == slenderness_TR, 'Pn_min_LB [ton]'].values[0]
    
    fig = go.Figure()
    
    fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})

    fig.add_trace(
        go.Scatter(
            x = df_C['Lc [m]'],
            y = df_C['Pn_min_LB [ton]'],
            hovertemplate = '',
            opacity = 0.0,
            showlegend = False,
            hoverinfo = 'skip',
        )
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_C['Pn_min_LB [ton]'],
            name = 'Pn min with LB',
            mode = 'lines',
            line = dict(color='firebrick', width=3,),
            customdata = df_C['Lc [m]'],
            hovertemplate =
            "Pn min LB: %{y:.2f} ton<br>Lc = %{customdata:.2f} m<extra></extra>",
        )
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_C['Pn_FB [ton]'],
            name = 'Pn FB without LB',
            mode = 'lines',
            line = dict(color='#FF6692',width=3, dash = 'dash'),
            hovertemplate =
            "Pn FB: %{y:.2f} ton<extra></extra>",
        )
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_C['Pn_FTB [ton]'],
            name = 'Pn FTB without LB',
            mode = 'lines',
            line = dict(color='#17BECF',width=3, dash = 'dash'),
            hovertemplate =
            "Pn FTB: %{y:.2f} ton<extra></extra>",
        )
    ),
    fig.add_trace(
        go.Scatter(
            x = [x_limit , x_limit],
            y = [0 , y_limit],
            mode = 'lines+text',
            line = dict(dash = 'dash',color = 'black',),
            text = [f'   <b>Lc/r transition = {slenderness_TR:.2f}<br>   Lc = {xx:.2f} m</b>'],
            textposition = 'top right',
            hoverinfo = 'skip',
            showlegend = False,
        )
    ),
    
    fig.data[0].update(xaxis='x2')

    fig.update_layout(
        xaxis2 = dict(title_text=xaxis2_title,
                      fixedrange=True,
                      showgrid=False,
                      range=[0, df_C['Lc [m]'].max()],
                     ),
    )
    
    fig.update_layout(
        xaxis = dict(title = xaxis_title, fixedrange=True, range=[0,x_data.max()],
                     showspikes = True, spikemode="toaxis+across", spikedash="dash", spikethickness=3, spikecolor='#868686',
                    ),
        yaxis = dict(title = 'Nominal Compressive Strength, Pn [ton]', fixedrange=True,
                     showspikes = True, spikemode="toaxis", spikedash="dash", spikethickness=3, spikecolor='#868686',
                    ),
        hovermode = 'x',
        showlegend = True,
        legend = dict(orientation="h", yanchor="bottom", y=1.25, xanchor="left", x=0.0),
        margin = dict(b=0, l=0, r=40),
    )
    
    fig.add_annotation(
        text = message,
        font = dict(size=14, color="black"),
        xref = "paper", yref="paper", x=0.05, y=0.05, showarrow=False
    )
    
    return fig
    
def flexural_cal(E = E_default, Fy = Fy_default, Cb = 1.0, Mservice_factor = 0.5):

    Lp = df['Lp [m]'][0]
    Lr = df['Lr [m]'][0]

    Myx = df['My x [ton-m]'][0]
    Mpx = df['Mp x [ton-m]'][0]
    rts = df['rts [cm]'][0]
    J = df['J [cm4]'][0]
    h0 = df['h0 [mm]'][0]
    
    h = df['d [mm]'][0] - 2*(df['tf [mm]'][0]+df['r [mm]'][0])
    tw = df['tw [mm]'][0]
    Sx = df['Sx [cm3]'][0]
    
    flange = df['Flexural: Flange Classification'][0]
    lambda_f = df['0.5bf/tf'][0]
    f_lambda_pf = df['f_lambda_pf'][0]
    f_lambda_rf = df['f_lambda_rf'][0]


    # F3.2: Compression Flange Local Buckling
    # a) For sections with noncompact flanges        
    if flange == 'Compact':
        Mn_CFLB = 999999999999999999999999
    elif flange == 'NonCompact':
        Mn_CFLB = Mpx - (Mpx-0.7*Myx)*((lambda_f-f_lambda_pf)/(f_lambda_rf-f_lambda_pf)) # ton-m
    elif flange == 'Slender': 
        kc = 4/(np.sqrt(h/tw))
        Mn_CFLB = 0.9*E*kc*Sx/(lambda_f**2)

    step = 0.05
    aa = np.arange(0, Lp, step)
    aa = np.append(aa,[Lp])
    bb = np.arange(np.ceil(Lp*10)/10,Lr,step)
    bb = np.append(bb,[Lr])
    cc = np.arange(np.ceil(Lr*10)/10,Lr+3.0+step,step)

    L = (np.concatenate((aa, bb, cc), axis=0))          
            
    Mn_YIELD_list = []
    Mn_LTB_list = []
    Mn_CFLB_list = []
    for Lb in L:
        Mn_YIELD_list.append(Mpx)
        Mn_CFLB_list.append(Mn_CFLB)
        if Lb <= Lp:
            Mn_LTB_list.append(Mpx)
        elif Lb > Lp and Lb <= Lr:
            Mrx = min( Mpx , Cb*(Mpx-(Mpx-0.7*Myx)*((Lb-Lp)/(Lr-Lp))) )
            Mn_LTB_list.append(Mrx)
        elif Lb > Lr:
            Fcr = (Cb*np.pi**2*E / (Lb*100/rts)**2) * np.sqrt(1+0.078*J*c/(Sx*h0/10)*(Lb*100/rts)**2) # ksc
            Mn_LTB_list.append(min( Mpx , Fcr*Sx/100/1000 ))
                       
    df_f = pd.DataFrame({'L':L,'Mn_Y':Mn_YIELD_list,'Mn_LTB':Mn_LTB_list,'Mn_CFLB':Mn_CFLB_list})
    
    if flange == 'NonCompact':
        df_f['Mn_min'] = df_f[['Mn_LTB','Mn_Y','Mn_CFLB']].min(axis=1)
    else:
        df_f['Mn_min'] = df_f[['Mn_LTB','Mn_Y']].min(axis=1)
        
    df_f['L/120'] = df_f['L']*1000/ 120
    df_f['L/180'] = df_f['L']*1000/ 180
    df_f['L/240'] = df_f['L']*1000/ 240
    df_f['L/360'] = df_f['L']*1000/ 360
    df_f['d_max'] = (5*8/384)*Mservice_factor*df_f['Mn_min']*(df_f['L']**2)*(10**10)/(E*df['Ix [cm4]'][0])

    return df_f

def flexural_plot(df_f):

    Lpx = df['Lp [m]'][0]
    Lrx = df['Lr [m]'][0]
    Mpx = df['Mp x [ton-m]'][0]
    Mrx = df['Mr x [ton-m]'][0]
    
    x_data = df_f['L']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['Mn_min'],
            name = 'Mn min',
            mode = 'lines',
            line = dict(color='firebrick', width=3,),
            hovertemplate =
            "Mn min: %{y:.2f} ton-m<extra></extra>",
        ), secondary_y=False,
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['Mn_Y'],
            name = 'Mn Yielding',
            mode = 'lines',
            line = dict(color='#17BECF', width=3, dash = 'dash',),
            hovertemplate =
            "Mn Yielding: %{y:.2f} ton-m<extra></extra>",
        ), secondary_y=False,
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['Mn_LTB'],
            name = 'Mn LTB',
            mode = 'lines',
            line = dict(color='#FF6692', width=3, dash = 'dash',),
            hovertemplate =
            "Mn LTB: %{y:.2f} ton-m<extra></extra>",
        ), secondary_y=False,
    ),

    if df['Flexural: Flange Classification'][0] == 'NonCompact':
        yy = df_f['Mn_CFLB']
        op = 1.0
        bo = True
        ho = 'all'
        hot = "Mn CFLB: %{y:.2f} ton-m<extra></extra>"
    else:
        yy = df_f['Mn_Y']
        op = 0.0
        bo = False
        ho = 'skip'
        hot = ''
        
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = yy,
            name = 'Mn CFLB',
            mode = 'lines',
            line = dict(color='#48ff9a', width=3, dash = 'dash',),
            hovertemplate = hot,
            opacity = op,
            showlegend = bo,
            hoverinfo = ho,
        ), secondary_y=False,
    ),
    
    
    fig.add_trace(
        go.Scatter(
            x = [Lpx , Lpx],
            y = [0 , Mpx],
            name = 'Lp',
            mode = 'lines+text',
            line = dict(dash = 'dash',color = 'black',),
            text = [f'   <b>Lp = {Lpx:.2f} m</b>'],
            textposition = 'bottom right',
            hoverinfo = 'skip',
            showlegend = False,
        ), secondary_y=False,
    ),
    fig.add_trace(
        go.Scatter(
            x = [Lrx , Lrx],
            y = [0 , Mrx],
            name = 'Lr',
            mode = 'lines+text',
            line = dict(dash = 'dash',color = 'black',),
            text = [f'   <b>Lr = {Lrx:.2f} m</b>'],
            textposition = 'bottom right',
            hoverinfo = 'skip',
            showlegend = False,
        ), secondary_y=False,
    ),
    
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['L/120'],
            name = 'L/120',
            mode = 'lines',
            line = dict(color='#8688f8', width=1),
            hovertemplate =
            "L/120: %{y:.2f} mm<extra></extra>",
            showlegend = False,
        ), secondary_y=True,
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['L/180'],
            name = 'L/180',
            mode = 'lines',
            line = dict(color='#8688f8', width=1),
            hovertemplate =
            "L/180: %{y:.2f} mm<extra></extra>",
            showlegend = False,
        ), secondary_y=True,
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['L/240'],
            name = 'L/240',
            mode = 'lines',
            line = dict(color='#8688f8', width=1),
            hovertemplate = "L/240: %{y:.2f} mm<extra></extra>",
            showlegend = False,
        ), secondary_y=True,
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['L/360'],
            name = 'L/360',
            mode = 'lines',
            line = dict(color='#8688f8', width=1),
            hovertemplate = "L/360: %{y:.2f} mm<extra></extra>",
            showlegend = False,
        ), secondary_y=True,
    ),
    fig.add_trace(
        go.Scatter(
            x = x_data,
            y = df_f['d_max'],
            name = 'Deflection MAX',
            mode = 'lines',
            line = dict(color='#5c5eab', width=3),
            hovertemplate = "MAX: %{y:.2f} mm<extra></extra>",
            showlegend = True,
        ), secondary_y=True,
    ),
        
    fig.update_layout(
        xaxis = dict(title = 'Unbraced Length, Lb [m]', fixedrange=True, range=[0,x_data.max()],
                     showspikes = True, spikemode="toaxis", spikedash="dash", spikethickness=3, spikecolor='#868686',
                    ),
        yaxis = dict(title = 'Nominal Flexural Strength, Mn [ton-m]', fixedrange=True,
                     showspikes = True, spikemode="toaxis", spikedash="dash", spikethickness=3, spikecolor='#868686',
                    ),
        hovermode = 'x',
        showlegend = True,
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin = dict(t=0, b=0, l=0, r=40),
        yaxis2 = dict(title_text='Deflection [mm]', fixedrange=True, tickmode="sync"),
    )

    return fig

def combined_load_plot(Mminor_ratio = 0.0):
    df_combined = pd.DataFrame({'Axial_ratio':np.arange(0,1.01,0.01)})
    df_combined.loc[df_combined['Axial_ratio'] >= 0.2, 'Mmajor_ratio'] = (1.0 - df_combined.loc[df_combined['Axial_ratio'] >= 0.2, 'Axial_ratio'])*9/8 - Mminor_ratio
    df_combined.loc[df_combined['Axial_ratio'] < 0.2, 'Mmajor_ratio'] = (1.0 - df_combined.loc[df_combined['Axial_ratio'] < 0.2, 'Axial_ratio']/2) - Mminor_ratio
    
    yy = df_combined.loc[df_combined['Axial_ratio'] == 0.2]['Axial_ratio'].to_list()[0]
    xx = df_combined.loc[df_combined['Axial_ratio'] == 0.2]['Mmajor_ratio'].to_list()[0]
    if xx < 0:
        xx = 0
    
    df_combined.drop(df_combined.loc[df_combined['Mmajor_ratio'] < 0].index, inplace = True)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x = df_combined['Mmajor_ratio'],
            y = df_combined['Axial_ratio'],
            name = 'Mn_min',
            mode = 'lines',
            line = dict(color='firebrick', width=3,),
            hovertemplate =
            "Axial_ratio: %{y:.2f}<br>" +
            "Mmajor_ratio: %{x:.2f}<br>" +
            f"Mminor_ratio: " + str(Mminor_ratio) +
            "<extra></extra>",
        ),        
    ),
    
    fig.add_trace(
        go.Scatter(
            x = [0 , xx, xx],
            y = [yy , yy, 0],
            name = 'Lr',
            mode = 'lines',
            line = dict(dash = 'dash',color = 'black', width=1,),
            hoverinfo = 'skip',
            showlegend = False,
        )
    ),
    
    fig.update_layout(
        xaxis = dict(title = 'Mrx / Mcx', fixedrange=True, range=[0, 1],
                     showspikes = True, spikemode="toaxis", spikedash="dash", spikethickness=3, spikecolor='#868686',
                    ),
        yaxis = dict(title = 'Pr / Pc', fixedrange=True, range=[0, 1],
                     showspikes = True, spikemode="toaxis", spikedash="dash", spikethickness=3, spikecolor='#868686',
                    ),
        hovermode = 'x',
        showlegend = False,
        xaxis_showspikes = True,
        yaxis_showspikes = True,
        margin = dict(t=0, b=0, l=0, r=40),       
    )
    
    fig.add_annotation(
        text = 'Mry / Mcy = '+str(Mminor_ratio),
        font = dict(size=14, color="black"),
        xref = "paper", yref="paper", x=0.05, y=0.05, showarrow=False
    )
    
    
    return fig

# initial calculation by default properties
section_classification()
section_classification_all()
df_C, slenderness_TR, message = compression_cal()
fig_compression = compression_plot(df_C, slenderness_TR, message)
df_f = flexural_cal()
fig_flexural = flexural_plot(df_f)
fig_combined = combined_load_plot()
    
# Create interactive components
E_input = pn.widgets.TextInput(name='E [kg/cm2]', value=str(E_default), width=210)
Fy_input = pn.widgets.TextInput(name='Fy [kg/cm2]', value=str(Fy_default), width=210)
section_select = pn.widgets.Select(name='Section', options=df_all.index.tolist(), value=section_default, width=210)
reset_button = pn.widgets.Button(name='Reset E & Fy to Default Value', width=210)

braced_minor = pn.widgets.Select(name='MINOR axis bracing', width=190,
                              options={'No Bracing': 1.0, '0.5L': 0.5, '1/3L': 1/3, '0.25L': 0.25, '0.1L': 0.1})
braced_major = pn.widgets.Select(name='MAJOR axis bracing', width=190,
                              options={'No Bracing': 1.0, '0.5L': 0.5, '1/3L': 1/3, '0.25L': 0.25, '0.1L': 0.1})
braced_torsion = pn.widgets.Select(name='Torsional bracing', width=190, options={'No Bracing': 1.0, '0.5L': 0.5, '1/3L': 1/3, '0.25L': 0.25, '': 0.1})

Cb = pn.widgets.FloatInput(name='Cb', start=1.0, step=0.05, value=1.0, width=285)

Mservice_factor = pn.widgets.FloatSlider(name='M service / Mn MIN', start=0.0, end=1.0, step=0.01, value=0.5, width=285)

Mminor_ratio = pn.widgets.FloatSlider(name='Mry / Mcy', start=0.0, end=1.0, step=0.01, value=0.0, width=570)

section_table = pn.widgets.DataFrame(width=380, text_align='left', row_height=25)
classification_table = pn.widgets.DataFrame(width=380, text_align='left', row_height=25)
flexural_table = pn.widgets.DataFrame(width=380, text_align='left', row_height=25)
shear_table = pn.widgets.DataFrame(width=380, text_align='left', row_height=25)
plotly_fig_compression = pn.pane.Plotly(fig_compression, width=570, height=400)
plotly_fig_flexural = pn.pane.Plotly(fig_flexural, width=570, height=400)
plotly_fig_combined = pn.pane.Plotly(fig_combined, width=570, height=400)

steel_table = pn.widgets.DataFrame(data, height=680, width=1210, frozen_columns=1, autosize_mode='fit_columns', text_align='right', row_height=25)

link1 = pn.widgets.Button(name='Steel Table (TIS)', width=210)
link2 = pn.widgets.Button(name='Section Analysis', width=210)

# Define callback functions
def update_table(event):
    global E, Fy
    try:
        E = float(E_input.value)
        Fy = float(Fy_input.value)
        if E <= 0 or Fy <= 0:
            raise ValueError
    except ValueError:
        E = E_default
        Fy = Fy_default
        
    section = section_select.value
    
    # Update classification based on the new E and Fy
    section_classification(E, Fy, section, Cb.value)
    
    # update data
    section_df = df.iloc[:, :22].T
    classification_df = df.iloc[:, 22:34].T
    flexural_df = df.iloc[:, 34:42].T
    shear_df = pd.concat([pd.DataFrame(df['h/tw']).T, df.iloc[:, 42:].T])
    section_table.value = pd.DataFrame(section_df)
    classification_table.value = pd.DataFrame(classification_df)
    flexural_table.value = pd.DataFrame(flexural_df)
    shear_table.value = pd.DataFrame(shear_df)
    
def update_table_all(event):
    global E, Fy
    try:
        E = float(E_input.value)
        Fy = float(Fy_input.value)
        if E <= 0 or Fy <= 0:
            raise ValueError
    except ValueError:
        E = E_default
        Fy = Fy_default
        
    section = section_select.value
    
    # Update classification based on the new E and Fy
    section_classification_all(E, Fy)
    
    steel_table.value = data

def reset_values(event):
    E_input.value = str(E_default)
    Fy_input.value = str(Fy_default)
    # section_select.value = df.index[0]
    update_table(event)
    update_table_all(event)
    
def update_compression(event):
    try:
        E = float(E_input.value)
        Fy = float(Fy_input.value)
        if E <= 0 or Fy <= 0:
            raise ValueError
    except ValueError:
        E = E_default
        Fy = Fy_default
    
    df_C, slenderness_TR, message = compression_cal(E, Fy, braced_minor.value, braced_major.value, braced_torsion.value)
    fig_compression = compression_plot(df_C, slenderness_TR, message)
    plotly_fig_compression.object = fig_compression
    
def update_flexural(event):
    try:
        E = float(E_input.value)
        Fy = float(Fy_input.value)
        if E <= 0 or Fy <= 0:
            raise ValueError
    except ValueError:
        E = E_default
        Fy = Fy_default
        
    df_f = flexural_cal(E, Fy, Cb.value, Mservice_factor.value)
    fig_flexural = flexural_plot(df_f)
    plotly_fig_flexural.object = fig_flexural
    
def update_combined_load(event):
    fig_combined = combined_load_plot(Mminor_ratio.value)
    plotly_fig_combined.object = fig_combined
    
def load_content1(event):
    template.main[0].objects = content1

def load_content2(event):
    template.main[0].objects = content2
    
# Register callback
section_select.param.watch(update_table, 'value')
E_input.param.watch(update_table, 'value')
Fy_input.param.watch(update_table, 'value')
Cb.param.watch(update_table, 'value')

section_select.param.watch(update_compression, 'value')
E_input.param.watch(update_compression, 'value')
Fy_input.param.watch(update_compression, 'value')
braced_minor.param.watch(update_compression, 'value')
braced_major.param.watch(update_compression, 'value')
braced_torsion.param.watch(update_compression, 'value')

section_select.param.watch(update_flexural, 'value')
E_input.param.watch(update_flexural, 'value')
Fy_input.param.watch(update_flexural, 'value')
Cb.param.watch(update_flexural, 'value')
Mservice_factor.param.watch(update_flexural, 'value')

Mminor_ratio.param.watch(update_combined_load, 'value')

reset_button.on_click(reset_values)

link1.on_click(load_content1)
link2.on_click(load_content2)

E_input.param.watch(update_table_all, 'value')
Fy_input.param.watch(update_table_all, 'value')

# Set initial state of the dashboard
default_section = section_select.value
update_table(default_section)
update_compression(default_section)
update_flexural(default_section)


# Instantiate the template with widgets displayed in the sidebar
template = pn.template.FastListTemplate(
    title='Steel Design (AISC360-16)',
    sidebar=[link1, link2, E_input, Fy_input, reset_button, section_select],
    accent_base_color="#e85eff",
    header_background="#be5eff",
    theme='dark',
    sidebar_width = 230,
    meta_author = 'KJ'
)
# Append a layout to the main area, to demonstrate the list-like API
page = pn.Column(sizing_mode='stretch_width')
template.main.append(page)

content2 = [
    pn.Column(
        pn.Row(
            pn.Column(
                '## Section Data',
                section_table,
            ),
            pn.Column(
                '## Section Classifications',
                classification_table,
            ),
            pn.Column(
                '## Compression Data',
                pn.pane.LaTeX(r"$\quad \phi=0.90 \quad, \quad\quad \Omega=1.67$"),
                '## Flexural Data',
                pn.pane.LaTeX(r"$\quad \phi=0.90 \quad, \quad\quad \Omega=1.67$"),
                flexural_table,
                '## Shear Data',
                pn.pane.LaTeX(r"$\quad \phi=1.00 \quad, \quad\quad \Omega=1.50$"),
                shear_table,
            ),
        ),
        pn.Row(
            pn.Column(
                '## Compressive Strength of Section',
                plotly_fig_compression,
                # x_button,
                pn.Row(braced_minor, braced_major, braced_torsion),
            ),
            pn.Column(
                '## Flexural Strength of Section (Major Axis)',
                plotly_fig_flexural,
                pn.Row(Cb, Mservice_factor),
                pn.Column(
                    pn.pane.Markdown(r"$$\text{Deflection Max, }\delta_{\max }=\frac{5 \omega L^4}{384 E I}$$"),
                    pn.pane.Markdown(r"$$\text{Service Moment, }M_{service}=\frac{w L^2}{8}$$"),
                    pn.pane.Markdown(r"$$\text{Yield }\delta_{\max }=\frac{5 \times 8 M_{service} L^2}{384 E I}$$"),
                )
            ),
        ),
        pn.Row(
            pn.Column(
                '## Combined Load',
                plotly_fig_combined,
                Mminor_ratio,
            ),
        ),

    ),    
]

content1 = [pn.Column(steel_table, pn.pane.Markdown(r'$$C_b = 1.0$$'))]

load_content1(content1)
load_content2(content2)

# template.show()
template.servable()
