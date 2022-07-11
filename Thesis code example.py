import math
import statistics
import os
import sys

import scipy.signal

import Run_parameters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.ndimage import correlate
from scipy.signal import peak_widths


import time
start_time = time.time()

slide_num = 0



#I am sorry about the poor quality of the comments and explanations, but the code is still a work in progress and needs some fine tuning.
# This file can be considered as more of a showcase of my abilities, since I am not expecting anyone to actually try to read all of the 1000 lines in the original file, not to mention actually understand what al of they do




#Data manipulation and drift fix

def Bgrhist(df,  All_df):
    if len(All_df) != 0:
        df.index = df.index + All_df.shape[0]
    All_df = All_df.append(df)

    return All_df

def PPC_mani(df, Limits, drift_fix):  # This function sorts the data and applies drift fix if needed
    df_all = df[cols]

    def clean_df(df_all):
        df_all.iloc[:,1:5] = df_all.iloc[:,1:5][(df_all.iloc[:,1:5]>Limits[0]) ]
        df_all.iloc[:,1:5] = df_all.iloc[:,1:5][(df_all.iloc[:,1:5]<Limits[1]) ]
        df_all = df_all.dropna(how='all')
        return df_all

    df_all = clean_df(df_all)

    def create_bins(df):
        bins_1 = [round(x) for x in pd.cut(df.TOF1, N_cutsy, retbins=True, right=False)[1]]
        bins_elx = [round(x) for x in pd.cut(df.elx, N_cutsx, retbins=True, right=False)[1]]
        return bins_1, bins_elx

    bins_1, bins_elx = create_bins(df_all)
    time_df, index_bins = Time_df(df_all, bins_1, int( custom_time_bins[1]))


    def apply_cor(df, cor_par):
        one_rot = time.time()
        print("Start  one_rot" )
        for  inde, par in enumerate(cor_par.index.to_list()):
            par = int(par)
            if cor_par.iloc[inde][0]>0:
                try:
                    df.iloc[par:int(cor_par.index.to_list()[inde+1]), 0:4] = df.iloc[par:int(cor_par.index.to_list()[inde+1]), 0:4].apply(lambda x: np.round(x-cor_par.iloc[inde][0]*x +cor_par.iloc[inde][1],0))
                except:
                    df.iloc[par:, 0:4] = df.iloc[par:, 0:4].apply(lambda x: np.round(x-cor_par.iloc[inde][0]*x +cor_par.iloc[inde][1],0))

        print(time.time()-one_rot)
        return df

     # applies drift fix
    if drift_fix and Run_parameters.sample==2:
        time_df, cor_par = Time_plot(time_df, False)
        df = apply_cor(df, cor_par)

    df_rand = df[df.con == 2]
    df_true = df[df.con == 1]

    df = clean_df(df)
     # Creates figure 3 in pdf
    time_df, index_bins = Time_df(df, bins_1, int( custom_time_bins[1]))


    df_rand = df_rand[cols]
    df_true = df_true[cols]

    df_true[df_true.iloc[:, 1:5] < Limits[0]] = np.nan
    df_true[df_true.iloc[:, 1:5] > Limits[1]] = np.nan
    df_true[df_true.iloc[:, 0] < Limits[2]] = np.nan
    df_true[df_true.iloc[:, 0] > Limits[3]] = np.nan

    # For the future, elx drift not in pdf
    elx_hist = df_true.elx.groupby([pd.cut(df_true.elx, bins=50, right=False), pd.cut(df_true.index, bins=index_bins, right=False)]).count().unstack()
    #Elx_time(elx_hist)


    return time_df, df_true, df_rand

# Makes the over time tof so, figure 3
def Time_df(df, TOF_bins, index_bins):
    time_tof = pd.DataFrame()
    now_time = time.time()
    index_bins = np.round(pd.cut(df.index, bins= index_bins, right = False, retbins=True)[1],0)

    #Pandas makes everythings so easy this creates a 2D histogram
    for i in df.columns[1:5]:
        time_tof = time_tof.add(df[i].groupby([pd.cut(df[i], bins=TOF_bins, right=False), pd.cut(df.index, bins=index_bins, right=False)]).count().unstack(),fill_value=0)


    print("Time df ", time.time() - now_time)

    time_tof.index = TOF_bins[:-1]
    time_tof.columns = index_bins[:-1]

    return time_tof, index_bins


# This function does the drift fix
def Fix_drift(time_tof, run_index):
    plot_lines = False

    Peaks_pos, time_tof = Get_gpeaks(time_tof, run_index) #Find peak positions
    cor_par = Peaks_fit(Peaks_pos, time_tof)

    #Draw peak positions to heatmap
    if plot_lines:
        time_qua = time_tof.T.sum()[time_tof.T.sum() > time_tof.T.sum().quantile(0.80)]  # positions of % peaks
        positions = [time_tof.index.get_loc(col) for col in time_qua.index]
        for k in range(len(time_qua)):
            plt.axhline(y=positions[k], linewidth=0.7)

    return time_tof, cor_par


# Function makes the slides mentioned in the pdf and and finds peaks and fits the peaks figures 4 and 5 are mad e by this function
def Get_gpeaks(time_tof, run_index):
    Slides = False

    def Ranges(nums):
        averages = []
        nums = [i for i in nums if i != 0]
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        lists = list(zip(edges, edges))

        return lists

    def Average_from_list(list):
        return statistics.mean(list)

    Norm_tof = time_tof.reset_index(drop = True )

    errors = 0
    fig = plt.figure(0)

    slide_peaks = pd.DataFrame(columns=Norm_tof.index, index=Norm_tof.columns)
    peaks_pos = pd.DataFrame(columns=Norm_tof.index, index=Norm_tof.columns)

    # checks every slice for peaks
    for slice, col in enumerate(Norm_tof.T.iterrows()):
        pos_ran, _= find_peaks(Norm_tof.iloc[:,slice], prominence=1000 )
        width = peak_widths(Norm_tof.iloc[:,slice], pos_ran, rel_height=0.5)
        pos_ran = list(zip(np.round(width[2],0).astype(int),np.ceil(width[3]).astype(int)+1))


        # Fits a function to each peak
        for index, peak in enumerate(pos_ran):

            pos_ave = round(Average_from_list(peak))
            mean, sigma = norm.fit(Norm_tof.iloc[peak[0] :peak[-1] , slice])
            #ok, sigma = norm.fit(Norm_tof.iloc[peak[0] :peak[-1] , slice])

            dx = pd.DataFrame(np.arange(200))
            x = Norm_tof.iloc[peak[0] :peak[-1]+1 , slice].index.to_list()
            dx.iloc[peak[0] :peak[-1] +1, 0] = x

            dy = pd.DataFrame(np.full((200, 1), 0))

            y = Norm_tof.iloc[peak[0]:peak[-1] +1, slice].values.flatten().tolist()
            dy.iloc[peak[0] :peak[-1] +1, 0] = y

            norm_max = Norm_tof.iloc[peak[0] :peak[-1] +1, slice].max()

            try:
                popt, pcov = curve_fit(Gaussian_funtion, dx.values.flatten().tolist(), dy.values.flatten().tolist(),
                                       p0=[norm_max* 1, pos_ave+0.1, 0.01 * sigma], maxfev=800, bounds=( [norm_max * 0.99, (pos_ave) - 2, sigma * 0.000001], [norm_max * 1.01, (pos_ave) + 3, 0.05 * sigma]))
            except:
                errors = errors + 1
                popt = [1, peak[0], 0.01]
                print("er")

            slide_peaks = slide_peaks.astype(object)
            slide_peaks.iloc[slice, pos_ave] = popt

            peaks_pos.iloc[slice, pos_ave] = popt[1]

    print("Amount of errors ", errors)
    peaks_pos = peaks_pos.dropna(axis=1, how='all')
    slide_peaks = slide_peaks.dropna(axis=1, how='all')

    def Slide_tof():
        ax = sns.lineplot(data=Norm_tof, x=Norm_tof.index, y=Norm_tof.iloc[0:, slide_num].values.flatten())
        plt.xticks(range(len( Norm_tof.index))[::10],   time_tof.index[::10])
        #colors = ["b", "g", "r","c","m", "y","k"]
        for col_num, col in enumerate(slide_peaks.T.iterrows()):
            plt.plot(np.arange(0, 200, 0.2), Gaussian_funtion(np.arange(0, 200, 0.2 ), *slide_peaks.iloc[slide_num, col_num]), 'ro:', label='fit', markersize=2)

        ax.set_title("This is slide n: " + str(slide_num) + "in set ")
    if Run_parameters.Peak_slides:
        Slide_show(Slide_tof, Norm_tof.shape[1])

    return peaks_pos, time_tof

#Function makes the plot using the peak positions and fits a line shown in figures 6 and 7
def Peaks_fit(Peaks_pos, time_tof):
    Peaks_pos  = Peaks_pos - Peaks_pos.quantile(0.3, axis = 0,  numeric_only =False)
    Peaks_pos = Peaks_pos.dropna()
    correction_fit =  pd.DataFrame(columns = ["a", "b"])

    for index in Peaks_pos.iterrows():
        popt, pcov = curve_fit(Linear_function,index[1].index, index[1].values, bounds=([-np.inf,-np.inf], [np.inf,np.inf]))
        correction_fit.loc[index[0]] = popt.tolist()

    def Slide_dif():
        ax = Peaks_pos.iloc[slide_num].plot()
        ax.set_title("This is slide n: " + str(slide_num))

        popt, pcov = curve_fit(Linear_function,  Peaks_pos.columns , Peaks_pos.iloc[slide_num].values)
        plt.plot(Peaks_pos.columns ,Linear_function(Peaks_pos.columns, *popt ))

    if Run_parameters.Fit_slides:
        Slide_show(Slide_dif, time_tof.shape[1])

    correction_fit[correction_fit.a<0.0005] = 0

    return correction_fit



#Make PEPICO shown in figure 1 and 2 does not plot it
def Newer_cuts(All_df, All_df_rand, bins_1, bins_elx):
    all_cuts = pd.DataFrame()
    all_r_cuts = pd.DataFrame()
    cuttime = time.time()
    print("Start  cuttime" )

    # Again pandas makes this so much easier, it would be very was too , but the dataformat forces a loop that makes it a lot slower (full file run time max 100 s  for 50 milloin datapoints)
    for i in All_df.columns[1:5]:
        all_cuts = all_cuts.add(All_df[i].groupby([pd.cut(All_df[i], bins=bins_1, right=False), pd.cut(All_df.elx, bins=bins_elx, right=False)]).count().unstack() , fill_value=0) # Create heatmap for 1-4 TOFs
        r_cuts = All_df_rand[i].groupby([pd.cut(All_df_rand[i], bins=bins_1, right=False)]).count()
        #TOF_cuts = TOF_cuts.add(cuts, fill_value=0)
        all_r_cuts = all_r_cuts.append(r_cuts)
    all_r_cuts = all_r_cuts.sum()

    print(time.time()-cuttime)
    # TOF_cuts.index = bins_1[:bins_1.shape[0] -1]
    elx_ind = All_df.elx.groupby([pd.cut(All_df.elx, bins=bins_elx, right=False)]).count()

    cuts = all_cuts.iloc[::-1]

    r_cuts = all_r_cuts.iloc[::-1]
    both_cuts = all_cuts.T.sum().add( r_cuts)

    print("All_cuts")

    # This loop removes the backgrounds so it is the difference between figure 1 and 2
    for index, i in cuts.T.iterrows():
        negative = int(elx_ind[index]) / All_df.shape[0] * r_cuts  # negative = i.sum() / r_cuts.sum() * r_cuts # rand or not rand


        cuts[index] = i - negative
    cuts[cuts < 0] = 0

    both_cuts.index = [round(x) for x in bins_1[:len(bins_1) - 1]]


    return cuts, both_cuts


#Plots both figure 3 and 8.
def Time_plot(time_tof, time_heatmap):
    TOF_bins = time_tof.shape[0]
    index_bins = time_tof.shape[1]

    run_index = custom_indexs[0]
    num_ind = custom_indexs[1]

    if drift_fix == True and time_heatmap ==False:
        time_tof, cor_par = Fix_drift(time_tof, num_ind)
        #time_tof = time_tof[::-1]
    else:
        cor_par = []

    # Plot numbers
    if time_heatmap:
        time_tof = time_tof[::-1]
        time_tof[time_tof<0]=0
        plt_scatter = True
        sns.heatmap(time_tof, cmap="gist_heat")

        if plt_scatter:
            place = 0
            plt.scatter(num_ind, np.full(run_index.shape[0], time_tof.shape[0] / 2))

            for i, label in enumerate(run_index):
                place = label * index_bins / run_index.sum() + place
                plt.annotate(run_index.index[i], (place, np.full(len(run_index), TOF_bins / 2)[i]), color="cyan")
        plt.xlim(27,48)
        plt.ylim(200,10)
        plt.show()
    return time_tof, cor_par


#Actually plots 2 D histograms 1 and 2
def plot_cuts(All_cuts):
    All_cuts.index = np.round(All_cuts.index,0)
    Norm_cuts = All_cuts
    Norm_cuts = Norm_cuts.pow(1 / 2)

    Norm_cuts = Norm_cuts[Norm_cuts>0]
    # Norm_cuts = (All_cuts-All_cuts.mean())/All_cuts.std()
    Norm_cuts = Norm_cuts.fillna(0)

    Norm_cuts = pd.DataFrame(Smoothing(Norm_cuts, 2), columns=All_cuts.columns.values, index=All_cuts.index.values)

    Norm_cuts = Norm_cuts[Norm_cuts>0]
    # Norm_cuts = (All_cuts-All_cuts.mean())/All_cuts.std()
    Norm_cuts = Norm_cuts.fillna(0)

    sns.heatmap(Norm_cuts, cmap="gist_heat")
    sns.color_palette("YlOrBr", as_cmap=True)

    #plt.ylim(Run_parameters.tof_mass[0], Run_parameters.tof_mass[1])

    plt.xlabel("Elx")
    plt.ylabel("Mass-to-charge")
    plt.show()
    return

#Smooths out both figures 2 times so the figure is clearer
def Smoothing(Norm_cuts, times):
    if times == 0:
        Norm_cuts = Norm_cuts.to_numpy()
    else:
        weights = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        for smooth in range(times):
            Norm_cuts = correlate(Norm_cuts, weights, mode="constant") / 9
    return Norm_cuts

# Not mentioned in the pdf, but of course there is calibration
def TOF_calibration(*molecules):
    import math
    mol1 = Molecules.loc[molecules[0]]
    mol2 =Molecules.loc[molecules[1]]


    C = (mol2.f_time-mol1.f_time)/(math.sqrt(mol2.mass_0)-math.sqrt(mol1.mass_0))
    T_0 = mol1.f_time- C* math.sqrt(mol1.mass_0)
    print("Parameter C ",C, "Parameter T ", T_0)
    return C, T_0

def Calib_bins(calib_cut, bins_1, bins_elx, calib_parameters, do):
    if do:
        calib_cut.index = [Calib_func(x) for x in bins_1][::-1][:len(bins_1) - 1]
        calib_cut.columns = [round(y**2 * calib_parameters[0] + calib_parameters[1] * y  + calib_parameters[2], 0) for y in bins_elx][:len(bins_elx) - 1]
    else:
        calib_cut.index = [round(x, 1) for x in bins_1][::-1][:len(bins_1) - 1]
        calib_cut.columns = [round(x, 1) for x in bins_elx][:len(bins_elx) - 1]
    return calib_cut

def peak_calib(TOF_cuts, Molecules):
    peaks, ok = find_peaks(TOF_cuts, prominence=peak_detec) # #42 275 ilman
    Molecules.f_time = TOF_cuts.index[peaks]
    print(Molecules.f_time)
    TOF_cuts= TOF_cuts.apply(np.sqrt)
    #Molecules.f_time.loc["vesi"] = TOF_cuts.index[peaks[2]]
    #Molecules.f_time.loc["näyte"] = TOF_cuts.index[peaks[-1]]
    add_mass = False
    if add_mass:
        Molecules.mass_0 = Molecules.f_time.apply(Calib_func)
        print(Molecules.mass_0)
    Molecules.to_excel(os.path.join(Other_dir,"Molecules.xlsx"))

    return


# Rest of the code is not included in the pdf, you can take a look, but it is not as clean.





#TOF
def Tof_read(Tofs, Sample):
    Mass_plot = 0
    for Num in Tofs:
        data = pd.read_csv(os.path.join(TOF_dir, str(Sample) + '_0' + str(Num) + '.txt'), header=None, delimiter="\t",
                           index_col=False)
        data.columns = ["F_time", "Intensity"]
        peaks, _ = find_peaks(data["Intensity"], height=150, distance=100, prominence=1)
        fig, ax = plt.subplots()
        #if (Mass_plot == 1):
        #    data["Mass"] = ((data["F_time"] - T_0) / C_0) ** 2

        print(data)
        ax.plot(data["Mass"], data["Intensity"])
        ax.plot(data["Mass"][peaks], data["Intensity"][peaks], "x")
        dm = DraggableMarker(ax, data, Calib_bins)
        plt.show()
    return

#Calibration
def calib_reads(File_num, Sample_type):
    dt = np.dtype([('TOF1', np.uint16), ('TOF2', np.uint16), ('TOF3', np.uint16),
                   ('TOF4', np.uint16), ('Not need1', np.uint16), ('Not need2', np.uint16), ('Not need3', np.uint16),
                   ('Not need4', np.uint16), ("elx", np.uint16
                                              ,), ('con', np.uint16)])
    Number_of = np.dtype([("elx", np.uint32,)])

    PPC_file = os.path.join(Calib_dir, str(Sample_type)  +"_0"+ str(File_num) + '.dat')  # Number of ions extraction
    records = np.fromfile(PPC_file, Number_of, offset=0)
    records = np.delete(records, [0])  # Yeets the number of ions

    records.tofile("test.dat")  # Converting the whole file
    records = np.fromfile("test.dat", dt, offset=0)
    df = pd.DataFrame(records)
    df = df[df.con == 1]
    df = df[df.elx >= 1000]
    df = df["elx"]

    return df

def Disp_elx(Disp_Number):
    N_bins = 250
    cal = [calib_reads(x, "DispEpass50") for x in Disp_Number]
    coeffs = pd.DataFrame(columns=["a","b","c"])
    #var_matrixs = pd.DataFrame(columns=["a"])
    var_matrixs = []

    for index, y in enumerate(cal):
        Histogram, bin_edges = np.histogram(y, bins=N_bins)  # make histo and find max peak
        peak_dist = bin_edges[Histogram.argmax()]
        bin_centres = (bin_edges[:-1]-10 + bin_edges[1:]+10) / 2
        p0 = [np.amax(Histogram), peak_dist, Histogram.std()]
        if p0[2]< 500:
            p0[2]= 600

        coeffs.loc[index,["a","b","c"]], var_matrix = curve_fit(Gaussian_funtion, bin_centres, Histogram, p0= p0, bounds =( [p0[0]*0.95, -np.inf, 500],  [np.inf, np.inf, 1700] ) )

    def Slide_curve():
        Histogram, bin_edges = np.histogram(cal[slide_num], bins=N_bins)  # make histo and find max peak
        peak_dist = bin_edges[Histogram.argmax()]
        bin_centres = (bin_edges[:-1]-10 + bin_edges[1:]+10) / 2
        p0 = [np.amax(Histogram), peak_dist, Histogram.std()]
        if p0[2]< 500:
            p0[2]= 600
        coeff, var_matrix = curve_fit(Gaussian_funtion, bin_centres, Histogram, p0= p0, bounds =( [p0[0]*0.95, -np.inf, 500],  [np.inf, np.inf, 1700] ) )  # curve_fit(gauss, bin_centres, peak, p0=p0)
        hist_fit = Gaussian_funtion(bin_centres, *coeff)
        cal[slide_num].hist( bins = N_bins)
        plt.plot(bin_centres, hist_fit, label='Fitted data')

    #Slide_show(Slide_curve, len(cal) )

    return coeffs, var_matrixs

def Disp_fit(Disp_nums):
    coeffs, err_coeffs = Disp_elx(Disp_nums)
    place = [5.49-10.5+int(x)*0.5 for x in Disp_nums]

    peak_spots = coeffs["b"].values
    popt, _ = curve_fit(Polynomial_function, peak_spots, place)
    a, b, c = popt

    print('y = %.15f * x^2 + %.10f * x + %.5f' % (a, b, c))
    Disp_plot = False
    if Disp_plot:
        plt.scatter(peak_spots, place)
        x_line = np.arange(min(peak_spots), max(peak_spots))
        y_line = Polynomial_function(x_line, *popt)
        plt.plot(x_line, y_line,  color='green')
        plt.legend()
        plt.show()
    #popt = [2.597*10**-10,0.000119, -3.7984]
    return popt

# Fitted functions
def Gaussian_funtion(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

def Polynomial_function(x,a, b, c):
    return a * x**2 + b * x  + c

def Linear_function(x, a, b):
    return a*x+b

def non(x):
    return x

def Calib_func(x):
    x = float(x)
    return  ((x- T_0)/C)**2

def m_to_T(x):
    x = float(x)
    return round(T_0+C*math.sqrt(x),0)

#PPC Begins
def Custom_bins(time_bins):
    PPC_df = pd.read_excel(os.path.join(Other_dir, "PPC_events.xlsx"), index_col=0)
    runs_df = PPC_df[Samples[PPC_Sample]].dropna()
    index_nums = runs_df[Analysed_runs].sum() / 100000
    run_index = runs_df[Analysed_runs]
    num_index = run_index.cumsum() * index_nums / run_index.sum()
    return [time_bins[0], index_nums],[run_index, num_index]

def PPC_read(File_num, Sample_type):
    dt = np.dtype([('TOF1', np.uint16), ('TOF2', np.uint16), ('TOF3', np.uint16),
                   ('TOF4', np.uint16), ('Not need1', np.uint16), ('Not need2', np.uint16), ('Not need3', np.uint16),
                   ('Not need4', np.uint16), ("elx", np.uint16), ('con', np.uint16)])
    Number_of = np.dtype([("elx", np.uint32,)])
    PPC_file = os.path.join(PPC_dir,
                            'PPC' + str(Sample_type) + '_0' + str(File_num[0]) + '.dat')  # Number of ions extraction
    if os.path.isfile(PPC_file):
        records = np.fromfile(PPC_file, Number_of, offset=0)
        Ionnumber = records[0][0]
        records = np.delete(records, [0])

        records.tofile("test.dat")  # Converting the whole file
        records = np.fromfile("test.dat", dt, offset=0)

        df = pd.DataFrame(records)
    else:
        print("Cant read file num", File_num)
        return
    df["Number"] = File_num[0]
    #print(df.TOF1)
    return df, Ionnumber

def Num_ch(numbers):
    for number in range(len(numbers)):
        if (numbers[number] < 10):
            numbers[number] = "0" + str(numbers[number])
        else:
            numbers[number] = str(numbers[number])
    return numbers






def Slide_show(to_plot, amount_slides):
    slide_num = 0
    Slideshow = plt.figure(0)

    def Scroll_forward(fig):
        global slide_num
        slide_num += 1
        slide_num %= amount_slides
        fig.clear()
        print("Figure", slide_num)
        to_plot()
        plt.draw()

    def Click_backwards(fig):
        global slide_num
        slide_num -= 1
        slide_num %= amount_slides
        fig.clear()
        print("Figure", slide_num)
        to_plot()
        plt.draw()

    to_plot()
    Slideshow.canvas.mpl_connect('scroll_event', lambda event: Scroll_forward(Slideshow))
    Slideshow.canvas.mpl_connect('button_press_event', lambda event: Click_backwards(Slideshow))
    plt.show()



#Molecules = pd.DataFrame(columns=['nimi', 'kaava', 'm_0', "f_time", "näyte"])

# Initial parameters
Samples = ["25Br4Niz", "4Br5Niz", "4Br2Niz"]
cols = ["elx", "TOF1", "TOF2", "TOF3", "TOF4", "Number"]

cal_bin = True
Limits = Run_parameters.Limits # TOF, elx #0 7500 22000

save_load = Run_parameters.save_load
drift_fix = Run_parameters.drift_fix

Over_time = Run_parameters.Over_time
Hist_plot = Run_parameters.Hist_plot
Dhisto = Run_parameters.Dhisto


peak_detec = Run_parameters.Peak_detec


# Tof
Do_TOF = False
if Do_TOF:
    TOF_dir = os.path.dirname("D:\Yliopisto\Gradu_edwin\Python\TOF/")
    TOF_sample = 0
    TOF_Number = [15]
    #Tof_read(Num_ch(TOF_Number), Samples[TOF_sample])


# Calibration
Do_Calibration = True
if Do_Calibration:
    Calib_dir = os.path.dirname("D:\Yliopisto\Gradu\Python\DispE/")
    Disp_nums = Num_ch(list(range(1,14)))
    calib_parameters = Disp_fit(Disp_nums)

# PPC
PPC_dir = os.path.dirname("D:\Yliopisto\Gradu\Python\PPC_data/")
PPC_Sample = Run_parameters.sample
N_cutsx = Run_parameters.N_cutsx
N_cutsy = Run_parameters.N_cutsy


runs_list = Run_parameters.runs_list
Analysed_runs = runs_list[PPC_Sample]

#Analysed_runs = np.array([8,9,10,11,12])
Other_dir = os.path.dirname("D:\Yliopisto\Gradu\Python\Other_files/"+Samples[PPC_Sample]+"/")

Molecules = pd.read_excel(os.path.join(Other_dir,"Molecules.xlsx"), header=0, index_col=0)


C, T_0 = TOF_calibration("vesi", "näyte")

re_calib = False

time_bins = Run_parameters.time_bins  #TOF, Time
custom_time_bins, custom_indexs = Custom_bins(time_bins)

if save_load[1]:
    if Over_time:
        time_df = pd.read_csv(os.path.join(Other_dir, Samples[PPC_Sample] + '_Time_df.csv'), header=0, index_col=0)
    if  Hist_plot:
        TOF_cuts = pd.read_csv(os.path.join(Other_dir, Samples[PPC_Sample] + '_TOF_cuts.csv'), header=0, index_col=0)["0"]
    if Dhisto:
        newer_cut = pd.read_excel(os.path.join(Other_dir, Samples[PPC_Sample] + '_cuts.xlsx'), header=0, index_col=0)

else:
    Runs = time.time()
    print("Start  Runs" )

    All_df = pd.DataFrame()
    All_df_rand = pd.DataFrame()

    for ind, run in enumerate(Analysed_runs):
        df, Events = PPC_read(Num_ch([run]), Samples[PPC_Sample])
        All_df = Bgrhist(df,  All_df)

    time_df, All_df_true, All_df_rand = PPC_mani(All_df, Limits, Run_parameters.drift_fix)
    print(time.time()-Runs)

    cuts = time.time()
    print("Start  cuts" )

    # TOF and elx bins
    _, bins_1 = pd.cut(All_df_true.TOF1, N_cutsy, retbins=True, right=False)
    bins_1 = [round(x) for x in bins_1 ]
    _, bins_elx = pd.cut(All_df_true.elx, N_cutsx, retbins=True, right=False)
    bins_elx = [round(x) for x in bins_elx ]
    #time_df, TOF_i = Time_df(All_df_mod, int(custom_time_bins[0]), int(custom_time_bins[1]))

    newer_cut, TOF_cuts = Newer_cuts(All_df_true, All_df_rand, bins_1, bins_elx)
    newer_cut = Calib_bins(newer_cut, bins_1, bins_elx, calib_parameters, cal_bin)

    if re_calib:
        peak_calib(TOF_cuts, Molecules)




    if save_load[0]:
        newer_cut.to_excel(os.path.join(Other_dir, Samples[PPC_Sample] + '_cuts.xlsx'))
        time_df.to_csv(os.path.join(Other_dir, Samples[PPC_Sample] + '_Time_df.csv'))
        TOF_cuts.to_csv(os.path.join(Other_dir, Samples[PPC_Sample] + '_TOF_cuts.csv'))


if Over_time:
    Time_plot(time_df, True)



def Cali_hist(all_cuts, molecules):
    fig, axe = plt.subplots()
    axe.plot(all_cuts)

    for molecule in  molecules:
        molecule = Molecules.loc[molecule]
        plt.scatter(molecule.f_time, all_cuts[molecule.f_time], color = "red")
        axe.text(molecule.f_time ,all_cuts.loc[molecule.f_time]," m/q="+str(molecule.mass_0 ))
        axe.text(molecule.f_time-50 ,all_cuts.loc[molecule.f_time]+50,  molecule.nimi+"\n"+ str(round(molecule.f_time,0))+" ns\n m/q="+str( molecule.mass_0))
        #if molecule.f_time_2>0:
        #     axe.text(molecule.f_time_2 ,TOF_cuts.loc[molecule.f_time_2]," m/q="+str(molecule.mass_0+1 ))
        #     plt.scatter(molecule.f_time_2, TOF_cuts[molecule.f_time_2], color = "red")
        if molecule.f_time_3>0:
            axe.text(molecule.f_time_3 ,all_cuts.loc[molecule.f_time_3]," m/q="+str(molecule.mass_0 ))
            plt.scatter(molecule.f_time_3, all_cuts[molecule.f_time_3], color = "red")

    plt.show()
    return

def check_limits(Limits, Molecules):
    Molecules = Molecules[Molecules.f_time>Limits[0]]
    Molecules = Molecules[Molecules.f_time<Limits[1]]
    return Molecules


if Hist_plot[0]:
    peaks = Tof_plot(TOF_cuts)
    #plt.show()

if Hist_plot[1] :
    Molecules = check_limits(Limits, Molecules)
    #Cali_hist(TOF_cuts, Molecules.index.to_list())


if Dhisto:
    plot_cuts(newer_cut)



