from __future__ import absolute_import, division, print_function, unicode_literals
import arcpy
import os

# Load required packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as interpolate
import pandas as pd
import math

class AnnoteFinder(object):
    """callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.

    Register this function like this:

    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotes, ax=None, xtol=.01, ytol=.01):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata)) / float(len(xdata))) / 2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata)) / float(len(ydata))) / 2
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return (math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def __call__(self, event):

        if event.inaxes:

            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                # print(event.xdata, event.ydata)
                for x, y, a in self.data:
                    # print(x, y, a)
                    if ((clickX - self.xtol < x < clickX + self.xtol) and
                            (clickY - self.ytol < y < clickY + self.ytol)):
                        annotes.append(
                            (self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, ax, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            t = ax.text(x, y, " - %s" % (annote), )
            m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)


# Define plotting functions hsvtorgb and piper
def hsvtorgb(H, S, V):
    """
    Converts hsv colorspace to rgb
    INPUT:
        H: [mxn] matrix of hue ( between 0 and 2pi )
        S: [mxn] matrix of saturation ( between 0 and 1 )
        V: [mxn] matrix of value ( between 0 and 1 )
    OUTPUT:
        R: [mxn] matrix of red ( between 0 and 1 )
        G: [mxn] matrix of green ( between 0 and 1 )
        B: [mxn] matrix of blue ( between 0 and 1 )
    """
    # conversion (from http://en.wikipedia.org/wiki/HSL_and_HSV)
    C = V * S
    Hs = H / (np.pi / 3)
    X = C * (1 - np.abs(np.mod(Hs, 2.0 * np.ones_like(Hs)) - 1))
    N = np.zeros_like(H)
    # create empty RGB matrices
    R = np.zeros_like(H)
    B = np.zeros_like(H)
    G = np.zeros_like(H)
    # assign values
    h = np.floor(Hs)
    # h=0
    R[h == 0] = C[h == 0]
    G[h == 0] = X[h == 0]
    B[h == 0] = N[h == 0]
    # h=1
    R[h == 1] = X[h == 1]
    G[h == 1] = C[h == 1]
    B[h == 1] = N[h == 1]
    # h=2
    R[h == 2] = N[h == 2]
    G[h == 2] = C[h == 2]
    B[h == 2] = X[h == 2]
    # h=3
    R[h == 3] = N[h == 3]
    G[h == 3] = X[h == 3]
    B[h == 3] = C[h == 3]
    # h=4
    R[h == 4] = X[h == 4]
    G[h == 4] = N[h == 4]
    B[h == 4] = C[h == 4]
    # h=5
    R[h == 5] = C[h == 5]
    G[h == 5] = N[h == 5]
    B[h == 5] = X[h == 5]
    # match values
    m = V - C
    R = R + m
    G = G + m
    B = B + m
    return (R, G, B)


def mol_convert(dat_piper):
    # Convert chemistry into plot coordinates
    gmol = np.array([40.078, 24.305, 22.989768, 39.0983, 61.01714, 60.0092, 35.4527, 96.0636])
    eqmol = np.array([2., 2., 1., 1., 1., 2., 1., 2.])
    n = dat_piper.shape[0]
    meqL = (dat_piper / gmol) * eqmol
    sumcat = np.sum(meqL[:, 0:4], axis=1)
    suman = np.sum(meqL[:, 4:8], axis=1)
    cat = np.zeros((n, 3))
    an = np.zeros((n, 3))
    cat[:, 0] = meqL[:, 0] / sumcat  # Ca
    cat[:, 1] = meqL[:, 1] / sumcat  # Mg
    cat[:, 2] = (meqL[:, 2] + meqL[:, 3]) / sumcat  # Na+K
    an[:, 0] = (meqL[:, 4] + meqL[:, 5]) / suman  # HCO3 + CO3
    an[:, 2] = meqL[:, 6] / suman  # Cl
    an[:, 1] = meqL[:, 7] / suman  # SO4
    return cat, an, meqL


def ion_in_cartesian(cat,an, offset=0.05):
    # Convert into cartesian coordinates
    h = 0.5 * np.tan(np.pi / 3)
    cat_x = 0.5 * (2 * cat[:, 2] + cat[:, 1])
    cat_y = h * cat[:, 1]
    an_x = 1 + 2 * offset + 0.5 * (2 * an[:, 2] + an[:, 1])
    an_y = h * an[:, 1]
    d_x = an_y / (4 * h) + 0.5 * an_x - cat_y / (4 * h) + 0.5 * cat_x
    d_y = 0.5 * an_y + h * an_x + 0.5 * cat_y - h * cat_x
    return cat_x, cat_y, an_x, an_y, d_x, d_y


def interpolate_colors(offset=0.05):

    h = 0.5 * np.tan(np.pi / 3)
    offsety = offset * np.tan(np.pi / 3)
    # create empty grids to interpolate to
    x0 = 0.5
    y0 = x0 * np.tan(np.pi / 6)
    X = np.reshape(np.repeat(np.linspace(0, 2 + 2 * offset, 1000), 1000), (1000, 1000), 'F')
    Y = np.reshape(np.repeat(np.linspace(0, 2 * h + offsety, 1000), 1000), (1000, 1000), 'C')
    H = np.nan * np.zeros_like(X)
    S = np.nan * np.zeros_like(X)
    V = np.nan * np.ones_like(X)
    A = np.nan * np.ones_like(X)

    # create masks for cation, anion triangle and upper and lower diamond
    ind_cat = np.logical_or(np.logical_and(X < 0.5, Y < 2 * h * X),
                            np.logical_and(X > 0.5, Y < (2 * h * (1 - X))))
    ind_an = np.logical_or(np.logical_and(X < 1.5 + (2 * offset), Y < 2 * h * (X - 1 - 2 * offset)),
                           np.logical_and(X > 1.5 + (2 * offset), Y < (2 * h * (1 - (X - 1 - 2 * offset)))))
    ind_ld = np.logical_and(
        np.logical_or(np.logical_and(X < 1.0 + offset, Y > -2 * h * X + 2 * h * (1 + 2 * offset)),
                      np.logical_and(X > 1.0 + offset, Y > 2 * h * X - 2 * h)),
        Y < h + offsety)
    ind_ud = np.logical_and(np.logical_or(np.logical_and(X < 1.0 + offset, Y < 2 * h * X),
                                          np.logical_and(X > 1.0 + offset, Y < -2 * h * X + 4 * h * (1 + offset))),
                            Y > h + offsety)
    ind_d = np.logical_or(ind_ld == 1, ind_ud == 1)

    # Hue: convert x,y to polar coordinates
    # (angle between 0,0 to x0,y0 and x,y to x0,y0)
    H[ind_cat] = np.pi + np.arctan2(Y[ind_cat] - y0, X[ind_cat] - x0)
    H[ind_cat] = np.mod(H[ind_cat] - np.pi / 6, 2 * np.pi)
    H[ind_an] = np.pi + np.arctan2(Y[ind_an] - y0, X[ind_an] - (x0 + 1 + (2 * offset)))
    H[ind_an] = np.mod(H[ind_an] - np.pi / 6, 2 * np.pi)
    H[ind_d] = np.pi + np.arctan2(Y[ind_d] - (h + offsety), X[ind_d] - (1 + offset))

    # Saturation: 1 at edge of triangle, 0 in the centre,
    # Clough Tocher interpolation, square root to reduce central white region
    xy_cat = np.array([[0.0, 0.0],
                       [x0, h],
                       [1.0, 0.0],
                       [x0, y0]])
    xy_an = np.array([[1 + (2 * offset), 0.0],
                      [x0 + 1 + (2 * offset), h],
                      [2 + (2 * offset), 0.0],
                      [x0 + 1 + (2 * offset), y0]])
    xy_d = np.array([[x0 + offset, h + offsety],
                     [1 + offset, 2 * h + offsety],
                     [x0 + 1 + offset, h + offsety],
                     [1 + offset, offsety],
                     [1 + offset, h + offsety]])
    z_cat = np.array([1.0, 1.0, 1.0, 0.0])
    z_an = np.array([1.0, 1.0, 1.0, 0.0])
    z_d = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    s_cat = interpolate.CloughTocher2DInterpolator(xy_cat, z_cat)
    s_an = interpolate.CloughTocher2DInterpolator(xy_an, z_an)
    s_d = interpolate.CloughTocher2DInterpolator(xy_d, z_d)
    S[ind_cat] = s_cat.__call__(X[ind_cat], Y[ind_cat])
    S[ind_an] = s_an.__call__(X[ind_an], Y[ind_an])
    S[ind_d] = s_d.__call__(X[ind_d], Y[ind_d])
    # Value: 1 everywhere
    V[ind_cat] = 1.0
    V[ind_an] = 1.0
    V[ind_d] = 1.0
    # Alpha: 1 everywhere
    A[ind_cat] = 1.0
    A[ind_an] = 1.0
    A[ind_d] = 1.0
    # convert HSV to RGB
    R, G, B = hsvtorgb(H, S ** 0.5, V)
    RGBA = np.dstack((R, G, B, A))
    return s_cat, s_an, s_d, RGBA


def piper(df_input):
    dat_piper = df_input[['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']].values
    cat, an, meqL = mol_convert(dat_piper)

    offset = 0.05
    offsety = offset * np.tan(np.pi / 3)
    h = 0.5 * np.tan(np.pi / 3)
    # create empty grids to interpolate to
    x0 = 0.5
    y0 = x0 * np.tan(np.pi / 6)

    s_cat, s_an, s_d, RGBA = interpolate_colors()

    cat_x, cat_y, an_x, an_y, d_x, d_y = ion_in_cartesian(cat, an)

    # calculate RGB triples for data points
    # hue
    hcat = np.pi + np.arctan2(cat_y - y0, cat_x - x0)
    hcat = np.mod(hcat - np.pi / 6, 2 * np.pi)
    han = np.pi + np.arctan2(an_y - y0, an_x - (x0 + 1 + (2 * offset)))
    han = np.mod(han - np.pi / 6, 2 * np.pi)
    hd = np.pi + np.arctan2(d_y - (h + offsety), d_x - (1 + offset))
    # saturation
    scat = s_cat.__call__(cat_x, cat_y) ** 0.5
    san = s_an.__call__(an_x, an_y) ** 0.5
    sd = s_d.__call__(d_x, d_y) ** 0.5
    # value
    v = np.ones_like(hd)
    # rgb
    cat = np.vstack((hsvtorgb(hcat, scat, v))).T
    an = np.vstack((hsvtorgb(han, san, v))).T
    d = np.vstack((hsvtorgb(hd, sd, v))).T
    return (dict(cat=cat, an=an, diamond=d, meq = meqL))


def check_nak(x):
    if x[0] == 0 and x[2] > 0:
        return x[2]
    else:
        return x[0] + x[1]


def fillMissing(df):
    fieldnames = [u'Na', u'K', u'NaK', u'Ca', u'Mg', u'Cl', u'HCO3', u'CO3', u'SO4']
    # fill in nulls with 0
    for col in df.columns:
        if col in fieldnames:
            for i in range(len(df)):
                if df.loc[i, col] is None or df.loc[i, col] == '' or np.isnan(df.loc[i, col]):
                    df.loc[i, col] = 0
        else:
            df.col = 0

    # add missing columns
    for name in fieldnames:
        if name in df.columns:
            pass
        else:
            print(name)
            df[name] = 0

    return df


def calc_totals(df1):

    anions = ['Cl', 'HCO3', 'CO3', 'SO4']
    cations = ['Na', 'K', 'Ca', 'Mg', 'NaK']
    df1['anions'] = 0
    df1['cations'] = 0
    df1['NaK_meqL'] = 0
    df1['NaK_meqL'] = df1[['Na_meqL', 'K_meqL', 'NaK_meqL']].apply(lambda x: check_nak(x), 1)
    df1['NaK'] = df1['NaK_meqL']*23.0
    meq_an = [an+'_meqL' for an in anions]
    meq_cat = [cat+'_meqL' for cat in cations]
    df1['anions'] = df1[meq_an].sum(axis=1)
    df1['cations'] = df1[meq_cat].sum(axis=1)

    df1['EC'] = df1['anions'] - df1['cations']
    df1['CBE'] = df1['EC'] / (df1['anions'] + df1['cations'])
    df1['maj_cation'] = df1[['Ca_meqL', 'Mg_meqL', 'Na_meqL', 'K_meqL']].idxmax(axis=1)
    df1['maj_cation'] = df1['maj_cation'].apply(lambda x: str(x)[:-5],1)
    df1['maj_anion'] = df1[['Cl_meqL', 'SO4_meqL', 'HCO3_meqL', 'CO3_meqL']].idxmax(axis=1)
    df1['maj_anion'] = df1['maj_anion'].apply(lambda x: str(x)[:-5],1)
    df1['water_type'] = df1[['maj_cation', 'maj_anion']].apply(lambda x: str(x[0]) + '-' + str(x[1]), 1)
    return df1

def get_field_names(table):
    read_descr = arcpy.Describe(table)
    field_names = []
    for field in read_descr.fields:
        field_names.append(field.name)
    field_names.remove('OBJECTID')
    return field_names

def get_spatial_reference(table):
    return arcpy.Describe(table).spatialReference


def table_to_pandas_dataframe(table, field_names=None, query=None, sql_sn=(None, None)):
    """
    Load data into a Pandas Data Frame for subsequent analysis.
    :param table: Table readable by ArcGIS.
    :param field_names: List of fields.
    :return: Pandas DataFrame object.
    """
    # if field names are not specified
    if not field_names:
        field_names = get_field_names(table)
    # create a pandas data frame
    df = pd.DataFrame(columns=field_names)

    # use a search cursor to iterate rows
    with arcpy.da.SearchCursor(table, field_names, query, sql_clause=sql_sn) as search_cursor:
        # iterate the rows
        for row in search_cursor:
            # combine the field names and row items together, and append them
            df = df.append(dict(zip(field_names, row)), ignore_index=True)

    # return the pandas data frame
    return df

def data_to_rgb(df_input):
    data = df_input
    rgb = piper(df_input)

    data['cat_hex'] = [mpl.colors.rgb2hex(i) for i in rgb['cat']]
    data['an_hex'] = [mpl.colors.rgb2hex(i) for i in rgb['an']]
    data['diamond_hex'] = [mpl.colors.rgb2hex(i) for i in rgb['diamond']]
    return data


def parameter(displayName, name, datatype, parameterType='Required', direction='Input', defaultValue=None):
    """The parameter implementation makes it a little difficult to quickly create parameters with defaults. This method
    prepopulates some of these values to make life easier while also allowing setting a default value."""
    # create parameter with a few default properties
    param = arcpy.Parameter(
        displayName=displayName,
        name=name,
        datatype=datatype,
        parameterType=parameterType,
        direction=direction)

    # set new parameter to a default value
    param.value = defaultValue

    # return complete parameter object
    return param


def linkAnnotationFinders(afs):
    for i in range(len(afs)):
        allButSelfAfs = afs[:i] + afs[i + 1:]
        afs[i].links.extend(allButSelfAfs)

# Load data
class PiperPlt(object):
    def __init__(self):
        self.chem_file = None
        self.chem_df = None
        self.plottitle = "Piper"
        self.spatref = None
        self.savedlayer = None
        self.wrkspace = None
        self.alphalevel = None
        self.scaletds = None
        self.usetds = None
        self.parm_matches = None

    def piper_plot(self):
        """
        Create a Piper plot
        INPUT:
            dat_piper: [nx8] matrix, chemical analysis in mg/L
                        order: Ca Mg Na K HCO3 CO3 Cl SO4
            plottitle: string with title of Piper plot
            alphalevel: transparency level of points. If 1, points are opaque
            color: boolean, use background coloring of Piper plot
        OUTPUT:
            figure with piperplot
            dictionary with:
                if color = False:
                    cat: [nx3] meq% of cations, order: Ca Mg Na+K
                    an:  [nx3] meq% of anions,  order: HCO3+CO3 SO4 Cl
                if color = True:
                    cat: [nx3] RGB triple cations
                    an:  [nx3] RGB triple anions
                    diamond: [nx3] RGB triple central diamond
        """

        # make a dict of header and assigned parameter names for matching later
        inputdata = [i[0] for i in self.parm_matches]
        tablehead = [i[1] for i in self.parm_matches]
        header_lookup = dict(zip(inputdata,tablehead))
        input_lookup = dict(zip(tablehead,inputdata))

        df_input = self.chem_df
        df_input = df_input.rename(columns = input_lookup)
        df_input = fillMissing(df_input)
        dat_piper = df_input[['Ca','Mg','Na','K','HCO3','CO3','Cl','SO4']].values

        data = data_to_rgb(df_input)

        newfile = os.path.dirname(self.chem_file) + '/with_hex_' + os.path.splitext(os.path.basename(self.chem_file))[0] + ".csv"

        cat, an, meqL = mol_convert(dat_piper)

        meq_df = pd.DataFrame(meqL, columns=['Ca_meqL', 'Mg_meqL', 'Na_meqL', 'K_meqL', 'HCO3_meqL',
                                     'CO3_meqL', 'Cl_meqL', 'SO4_meqL'])
        data = pd.concat([data,meq_df], axis=1)
        data = calc_totals(data)

        # https://pubs.usgs.gov/wri/1986/4124/report.pdf
        # http://inside.mines.edu / ~epoeter / _GW / resultsNOV03.pdf


        data['Alkcalc'] = data[['HCO3','CO3']].apply(lambda x: x[0]+2*x[1],1)
        data['TDScalc'] = data[['Ca','Mg','NaK','Cl','SO4','Alkcalc']].apply(lambda x: np.sum(x[0:-1])+0.6*x[-1],1)


        data.to_csv(newfile, index_label='ID')

        mintds = np.min(data['TDScalc'].values)
        maxtds = np.max(data['TDScalc'].values)
        arcpy.AddMessage(mintds)
        arcpy.AddMessage(maxtds)

        if self.scaletds:
            tds_calc_norm = data['TDScalc'].apply(lambda x: (x-mintds)/(maxtds-mintds)*self.scaletds,1)
        else:
            tds_calc_norm = data['TDScalc'].apply(lambda x: (x-mintds)/(maxtds-mintds)*10,1)

        # Basic shape of piper plot
        plottitle = self.plottitle
        alphalevel = self.alphalevel


        offset = 0.05
        offsety = offset * np.tan(np.pi / 3)
        h = 0.5 * np.tan(np.pi / 3)
        ltriangle_x = np.array([0, 0.5, 1, 0])
        ltriangle_y = np.array([0, h, 0, 0])
        rtriangle_x = ltriangle_x + 2 * offset + 1
        rtriangle_y = ltriangle_y
        diamond_x = np.array([0.5, 1, 1.5, 1, 0.5]) + offset
        diamond_y = h * (np.array([1, 2, 1, 0, 1])) + (offset * np.tan(np.pi / 3))

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['svg.fonttype'] = 'none'

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', frameon=False, xticks=[], yticks=[])
        ax.plot(ltriangle_x, ltriangle_y, '-k')
        ax.plot(rtriangle_x, rtriangle_y, '-k')
        ax.plot(diamond_x, diamond_y, '-k')

        cat_x, cat_y, an_x, an_y, d_x, d_y = ion_in_cartesian(cat, an)
        s_cat, s_an, s_d, RGBA = interpolate_colors()

        # visualise
        plt.imshow(RGBA,
                   origin='lower',
                   aspect=1.0,
                   extent=(0, 2 + 2 * offset, 0, 2 * h + offsety))
        # labels and title
        plt.title(plottitle)
        plt.text(-offset, -offset, u'$Ca^{2+}$', horizontalalignment='left', verticalalignment='center')
        plt.text(0.5, h + offset, u'$Mg^{2+}$', horizontalalignment='center', verticalalignment='center')
        plt.text(1 + offset, -offset, u'$Na^+ + K^+$', horizontalalignment='right', verticalalignment='center')
        plt.text(1 + offset, -offset, u'$HCO_3^- + CO_3^{2-}$', horizontalalignment='left', verticalalignment='center')
        plt.text(1.5 + 2 * offset, h + offset, u'$SO_4^{2-}$', horizontalalignment='center', verticalalignment='center')
        plt.text(2 + 3 * offset, -offset, u'$Cl^-$', horizontalalignment='right', verticalalignment='center')


        # plot data

        if self.usetds:
            pass
        else:
            tds_calc_norm = 1

        if 'ID' not in data.columns:
            data['ID'] = data.index

        ann = ["{:} Ca: {:.2f}\nMg: {:.2f}\nNaK: {:.2f}\nCl: {:.2f}\nSO4: {:.2f}\nHCO3: {:.2f}\nCO3: {:.2f}".format(data.loc[i,'ID'],
                                                                                               data.loc[i,'Ca'],
                                                                                           data.loc[i,'Mg'],
                                                                                           data.loc[i,'NaK'],
                                                                                           data.loc[i,'Cl'],
                                                                                           data.loc[i,'SO4'],
                                                                                           data.loc[i,'HCO3'],
                                                                                           data.loc[i,'CO3']) for i in data.index]


        #http://scipy-cookbook.readthedocs.io/items/Matplotlib_Interactive_Plotting.html

        plt.scatter(cat_x, cat_y, s = tds_calc_norm, alpha=alphalevel)
        af1 = AnnoteFinder(cat_x, cat_y, ann)
        plt.connect('button_press_event', af1)
        plt.scatter(an_x, an_y, s = tds_calc_norm, alpha=alphalevel)
        af2 = AnnoteFinder(an_x, an_y, ann)
        plt.connect('button_press_event', af2)
        plt.scatter(d_x, d_y, alpha=alphalevel, s = tds_calc_norm )
        af3 = AnnoteFinder(d_x, d_y, ann)
        plt.connect('button_press_event', af3)
        linkAnnotationFinders([af1, af2, af3])

        plotfile = os.path.dirname(self.chem_file) + '/piperplot.pdf'
        plt.savefig(plotfile)


        arcpy.AddMessage(newfile)


        #arcpy.env.workspace = os.path.dirname(self.chem_file)
        # sr = arcpy.SpatialReference(self.spatref)
        # Make the XY event layer...
        arcpy.MakeXYEventLayer_management(newfile, "X", "Y", "in_memory", self.spatref, "")

        # Print the total rows
        arcpy.AddMessage(arcpy.GetCount_management("in_memory"))

        # Save to a layer file
        arcpy.CopyFeatures_management("in_memory", self.savedlayer, "", "0", "0", "0")

        plt.show()

        # select by location
        # myPt = arcpy.Point((df.extent.XMax + df.extent.XMin) / 2, (df.extent.YMax + df.extent.YMin) / 2)
        #myPtGeometry = arcpy.PointGeometry(myPt)
        #arcpy.SelectLayerByLocation_management(DOQQImport, "COMPLETELY_CONTAINS", myPtGeometry)


class Toolbox(object):
    def __init__(self):
        self.label = "PeetersPiper"
        self.alias = "PeetersPiper"

        # List of tool classes associated with this toolbox
        self.tools = [PiperTable,PiperTableFromLayer]

class PiperTable(object):
    def __init__(self):
        self.label = "Plot Piper and Points from CSV"
        self.description = """Create Color Table From File """
        self.canRunInBackground = False
        self.parameters = [
            parameter("Chemistry Data", "chem_file", "DEFile"),
            parameter("Spatial Reference", "spatref", "GPSpatialReference"),
            parameter("Matches for Table Headers", "head_names", 'GPValueTable'),
            parameter("Plot Title","plottitle","GPString",parameterType="Optional"),
            parameter("Alpha Level", "alphalevel", "GPDouble",),
            parameter("Use TDS for Plot?", "usetds", "GPBoolean", parameterType="Optional"),
            parameter("TDS Scale Factor for Plot","scaletds", "GPDouble",parameterType="Optional"),
            parameter("Layer output location", "savedlayer", "DEFeatureClass", direction="Output")
        ]
        self.parameters[0].filter.list = ['csv']
        self.parameters[2].columns = [['GPString', 'Plot Parameter'], ['GPString', 'Table Column']]
        self.parameters[3].value = 'Piper Plot'
        self.parameters[4].value = 1.0
        self.parameters[6].value = 50.0

    def getParameterInfo(self):
        """Define parameter definitions; http://joelmccune.com/lessons-learned-and-ideas-for-python-toolbox-coding/"""
        return self.parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation is performed.
        This method is called whenever a parameter has been changed."""
        if parameters[1].value and arcpy.Exists(parameters[0].value):
            if not parameters[2].altered:
                # use a search cursor to iterate rows
                csvtable = pd.read_csv(parameters[0].valueAsText)
                table_cols = list(csvtable.columns)
                parm_fields = ['Ca','Mg','Na','NaK','SO4','Cl','HCO3','CO3','X','Y','ID']
                vtab = []
                for parm in parm_fields:
                    if parm in table_cols:
                        vtab.append([parm, parm])
                    else:
                        vtab.append([parm, None])

                parameters[2].values = vtab

                parameters[2].filters[1].list = sorted(table_cols)

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        pplot = PiperPlt()
        pplot.chem_file = parameters[0].valueAsText
        pplot.chem_df = pd.read_csv(parameters[0].valueAsText)
        pplot.spatref = parameters[1].valueAsText
        pplot.parm_matches = parameters[2].value

        pplot.plottitle = parameters[3].valueAsText
        pplot.alphalevel = parameters[4].valueAsText
        pplot.usetds = parameters[5].value
        pplot.scaletds = parameters[6].value
        pplot.savedlayer = parameters[7].valueAsText
        pplot.piper_plot()
        return

class PiperTableFromLayer(object):
    def __init__(self):
        self.label = "Plot Piper from Layer"
        self.description = """Create Color Table From File """
        self.canRunInBackground = False
        self.parameters = [
            parameter("Input Feature", "in_layer", "GPFeatureLayer"),
            parameter("Matches for Table Headers", "head_names", 'GPValueTable'),
            parameter("Plot Title","plottitle","GPString",parameterType="Optional"),
            parameter("Alpha Level", "alphalevel", "GPDouble",),
            parameter("Use TDS for Plot?", "usetds", "GPBoolean", parameterType="Optional"),
            parameter("TDS Scale Factor for Plot","scaletds", "GPDouble",parameterType="Optional"),
            parameter("Layer output location", "savedlayer", "DEFeatureClass", direction="Output")
        ]
        self.parameters[0].filter.list = ['csv']
        self.parameters[1].columns = [['GPString', 'Plot Parameter'], ['GPString', 'Table Column']]
        self.parameters[2].value = 'Piper Plot'
        self.parameters[3].value = 1.0
        self.parameters[5].value = 50.0

    def getParameterInfo(self):
        """Define parameter definitions; http://joelmccune.com/lessons-learned-and-ideas-for-python-toolbox-coding/"""
        return self.parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal validation is performed.
        This method is called whenever a parameter has been changed."""
        if parameters[0].value and arcpy.Exists(parameters[0].value):
            if not parameters[1].altered:
                # use a search cursor to iterate rows
                csvtable = table_to_pandas_dataframe(parameters[0].valueAsText)
                table_cols = list(csvtable.columns)
                parm_fields = ['Ca','Mg','Na','NaK','SO4','Cl','HCO3','CO3','X','Y','ID']
                vtab = []
                for parm in parm_fields:
                    if parm in table_cols:
                        vtab.append([parm, parm])
                    else:
                        vtab.append([parm, None])

                parameters[1].values = vtab

                parameters[1].filters[1].list = sorted(table_cols)

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        pplot = PiperPlt()
        pplot.chem_file = parameters[0].valueAsText
        pplot.chem_df = table_to_pandas_dataframe(parameters[0].valueAsText)
        pplot.spatref = get_spatial_reference(parameters[0].valueAsText)
        pplot.parm_matches = parameters[1].value
        pplot.plottitle = parameters[2].valueAsText
        pplot.alphalevel = parameters[3].valueAsText
        pplot.usetds = parameters[4].value
        pplot.scaletds = parameters[5].value
        pplot.savedlayer = parameters[6].valueAsText
        pplot.piper_plot()
        return