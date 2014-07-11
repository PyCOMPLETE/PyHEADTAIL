from matplotlib import rc, rcdefaults
import pylab as pl
from colorsys import hsv_to_rgb


def mystyle(fontsz=16):
	
	font = {#'family' : 'normal',
			#'weight' : 'bold',
			'size'   : fontsz}
	print fontsz
	rcdefaults()
	rc('font', **font)
	
def mystyle_arial(fontsz=16, dist_tick_lab=10):
	
	rcdefaults()
	rc('font',**{'family':'sans-serif','sans-serif':['arial'], 'size':fontsz})
	rc(('xtick.major','xtick.minor','ytick.major','ytick.minor'), pad=dist_tick_lab)
	


def sciy():
	pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y') 
	
def scix():
	pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='x') 


def colorprog(i_prog, Nplots, v1 = .9, v2 = 1.):
	return hsv_to_rgb(float(i_prog)/float(Nplots), v1, v2)
