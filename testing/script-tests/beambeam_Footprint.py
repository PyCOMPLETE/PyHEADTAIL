import numpy as np
import csv,time
#import matplotlib.pyplot as plt

def getStringRep(data,nAmpl,nAngl):
    retVal = 'TMPNAME,'+str(nAmpl)+',1,<'+str(data['TUNX'][0])+';'+str(data['TUNY'][0])+'>';
    for N in range(1,nAmpl):
        retVal = retVal + ','+str(nAngl);
        for M in range(nAngl):
            retVal = retVal + ',<' + str(data['TUNX'][1+(N-1)*nAngl+M]) + ';' + str(data['TUNY'][1+(N-1)*nAngl+M])+'>';            
    return retVal;

def parseDynapTune(fileName,nAmpl,nAngl):
    reader = csv.reader(open(fileName, 'r'), delimiter=' ');
    headers = 0;
    data = {};
    names = [];
    for row in reader:
        if len(row)>0:
            if len(row[0])>0:
                if row[0][0] == '@':
                    headers = headers + 1;
                elif row[0][0] == '*':
                    for i in np.arange(1,len(row)):
                        tmpName = row[i].strip();
                        if len(tmpName)>0:
                            names.append(tmpName);
                            data[tmpName] = [];
            else:
                dataCount = 0;
                for i in np.arange(len(row)):
                    tmpValue = row[i].strip();
                    if len(tmpValue) > 0:
                        try:
                            value = float(tmpValue);
                            data[names[dataCount]].append(value);
                        except ValueError:
                            data[names[dataCount]].append(tmpValue);
                        dataCount = dataCount + 1;
    return getStringRep(data,nAmpl,nAngl);

class Footprint:
    _tunes = []; #tunes[nampl][nangl]['H'/'V']
    _nampl = 0;
    _maxnangl = 0;
    _name = 'noName';
    _dSigma = 1.0;

    #file pattern:
    #name,amplitudeCount,angleCount,<tunex;tuney>,<...;...>,...,angleCount,...
    #TODO make robust
    def __init__(self,stringRep,dSigma=1.0):
        self._dSigma = dSigma;
        self._tunes = []; #tunes[nampl][nangl]['H'/'V']
        self._nampl = 0;
        self._maxnangl = 0;
        self._name = 'noName';
        items = stringRep.strip().split(',');
        self._name = items[0];
        self._nampl = int(items[1]);
        current = 2;
        for i in np.arange(self._nampl):
            self._tunes.append([]);
            nangl = int(items[current]);
            if nangl > self._maxnangl:
                self._maxnangl = nangl;
            current = current + 1;
            for j in np.arange(nangl):
                self._tunes[i].append([]);
                tunesString = items[current].lstrip('<').rstrip('>').split(';');
                tunes = {};
                #print tunesString
                tunes['H'] = float(tunesString[0]);
                tunes['V'] = float(tunesString[1]);
                self._tunes[i][j] = tunes;
                current = current + 1;
    def getName(self):
        return self._name;
    def getNbAmpl(self):
        return self._nampl;
    def getNbAngl(self):
        return self._maxnangl;
    def getTunes(self,ampl,angl):
        return self._tunes[ampl][angl];
    def getHTune(self,ampl,angl):
        if angl >= len(self._tunes[ampl]) and angl < self._maxnangl:
            return self._tunes[ampl][len(self._tunes[ampl])-1]['H'];
        else:
            return self._tunes[ampl][angl]['H'];
    def getVTune(self,ampl,angl):
        if angl >= len(self._tunes[ampl]) and angl < self._maxnangl:
            return self._tunes[ampl][len(self._tunes[ampl])-1]['V'];
        else:
            return self._tunes[ampl][angl]['V'];
    def getPlottable(self):
        lines = [];
        lines.append([]);lines.append([]);
        for i in np.arange(0,self._nampl-1,2):
            for j in np.arange(self._maxnangl):
                lines[0].append(self.getHTune(i,j));
                lines[1].append(self.getVTune(i,j));
            for j in np.arange(self._maxnangl-1,-1,-1):
                lines[0].append(self.getHTune(i+1,j));
                lines[1].append(self.getVTune(i+1,j));
        if self._nampl%2 == 0:
            for j in np.arange(0,self._maxnangl-1,2):
                for i in np.arange(self._nampl-1,-1,-1):
                    lines[0].append(self.getHTune(i,j));
                    lines[1].append(self.getVTune(i,j));
                for i in np.arange(0,self._nampl,1):
                    lines[0].append(self.getHTune(i,j+1));
                    lines[1].append(self.getVTune(i,j+1));
            if self._maxnangl%2 != 0:
                for i in np.arange(self._nampl-1,-1,-1):
                    lines[0].append(self.getHTune(i,self._maxnangl-1));
                    lines[1].append(self.getVTune(i,self._maxnangl-1));
                lines[0].append(self.getHTune(0,self._maxnangl-2));
                lines[1].append(self.getVTune(0,self._maxnangl-2));
        else:
            for j in np.arange(self._maxnangl):
                lines[0].append(self.getHTune(self._nampl-1,j));
                lines[1].append(self.getVTune(self._nampl-1,j));
            for j in np.arange(self._maxnangl-1,-1,-2):
                for i in np.arange(self._nampl-1,-1,-1):
                    lines[0].append(self.getHTune(i,j));
                    lines[1].append(self.getVTune(i,j));
                for i in np.arange(0,self._nampl,1):
                    lines[0].append(self.getHTune(i,j-1));
                    lines[1].append(self.getVTune(i,j-1));
            #if self._maxnangl%2 != 0:
            #    for i in np.arange(self._nampl-1,-1,-1):
            #        lines[0].append(self.getHTune(i,0));
            #        lines[1].append(self.getVTune(i,0));
            #    lines[0].append(self.getHTune(0, 1));
            #    lines[1].append(self.getVTune(0, 1));
        return lines;

    def draw(self,fig,lines):
        fig.clear();
        plt.plot(lines[0],lines[1],'-b');
        plt.scatter([lines[0][-1]],[lines[1][-1]],marker='o',color='r');
        plt.draw();
        time.sleep(0.1);

    def drawPlottable(self):
        lines = [];
        lines.append([]);lines.append([]);
        fig = plt.figure(0);
        plt.ion();
        for i in np.arange(0,self._nampl-1,2):
            for j in np.arange(self._maxnangl):
                lines[0].append(self.getHTune(i,j));
                lines[1].append(self.getVTune(i,j));
                self.draw(fig,lines)
            for j in np.arange(self._maxnangl-1,-1,-1):
                lines[0].append(self.getHTune(i+1,j));
                lines[1].append(self.getVTune(i+1,j));
                self.draw(fig,lines)
        if self._nampl%2 == 0:
            for j in np.arange(0,self._maxnangl-1,2):
                for i in np.arange(self._nampl-1,-1,-1):
                    lines[0].append(self.getHTune(i,j));
                    lines[1].append(self.getVTune(i,j));
                    self.draw(fig,lines)
                for i in np.arange(0,self._nampl,1):
                    lines[0].append(self.getHTune(i,j+1));
                    lines[1].append(self.getVTune(i,j+1));
                    self.draw(fig,lines)
            if self._maxnangl%2 != 0:
                for i in np.arange(self._nampl-1,-1,-1):
                    lines[0].append(self.getHTune(i,self._maxnangl-1));
                    lines[1].append(self.getVTune(i,self._maxnangl-1));
                    self.draw(fig,lines)
                lines[0].append(self.getHTune(0,self._maxnangl-2));
                lines[1].append(self.getVTune(0,self._maxnangl-2));
                self.draw(fig,lines)
        else:
            for j in np.arange(self._maxnangl):
                lines[0].append(self.getHTune(self._nampl-1,j));
                lines[1].append(self.getVTune(self._nampl-1,j));
                self.draw(fig,lines)
            for j in np.arange(self._maxnangl-1,-1,-2):
                for i in np.arange(self._nampl-1,-1,-1):
                    lines[0].append(self.getHTune(i,j));
                    lines[1].append(self.getVTune(i,j));
                    self.draw(fig,lines)
                for i in np.arange(0,self._nampl,1):
                    lines[0].append(self.getHTune(i,j-1));
                    lines[1].append(self.getVTune(i,j-1));
                    self.draw(fig,lines)
            #if self._maxnangl%2 != 0:
            #    for i in np.arange(self._nampl-1,-1,-1):
            #        lines[0].append(self.getHTune(i,0));
            #        lines[1].append(self.getVTune(i,0));
            #        self.draw(fig,lines)
            #    lines[0].append(self.getHTune(0, 1));
            #    lines[1].append(self.getVTune(0, 1));
            #    self.draw(fig,lines)
        return lines;
    
    def getIntermediateTunes(self,ampl,angl):
        if angl > np.pi/2.0:
            print('ERROR angle ',angl,' is larger than pi/2');
        ampl0 = int(ampl/self._dSigma);
        if ampl0 > self.getNbAmpl():
            print('Error, amplitude is too large');
        tmpAngl = 2.0*(self.getNbAngl()-1)*angl/np.pi;
        angl0 = int(tmpAngl);
        d1 = ampl/self._dSigma - ampl0;
        d2 = tmpAngl - angl0;
        if d1==0 and d2 == 0:
            tuneX = self.getHTune(ampl0, angl0);
            tuneY = self.getVTune(ampl0, angl0);
        elif d1 == 0:
            tuneX = (self.getHTune(ampl0,angl0)*(1.0-d2) + self.getHTune(ampl0,angl0+1)*d2);    
            tuneY = (self.getVTune(ampl0,angl0)*(1.0-d2) + self.getVTune(ampl0,angl0+1)*d2);    
        elif d2 == 0:
            tuneX = (self.getHTune(ampl0,angl0)*(1.0-d1) + self.getHTune(ampl0+1,angl0)*d1);    
            tuneY = (self.getVTune(ampl0,angl0)*(1.0-d1) + self.getVTune(ampl0+1,angl0)*d1);           
        else:
            tuneX = (self.getHTune(ampl0,angl0)*(1.0-d1)*(1.0-d2) + self.getHTune(ampl0+1,angl0)*d1*(1.0-d2) + self.getHTune(ampl0,angl0+1)*d2*(1.0-d1) + self.getHTune(ampl0+1,angl0+1)*d1*d2);    
            tuneY = (self.getVTune(ampl0,angl0)*(1.0-d1)*(1.0-d2) + self.getVTune(ampl0+1,angl0)*d1*(1.0-d2) + self.getVTune(ampl0,angl0+1)*d2*(1.0-d1) + self.getVTune(ampl0+1,angl0+1)*d1*d2);    
        return [tuneX,tuneY];
    
    def getTunesForAmpl(self,qx,qy):
        ampl = np.sqrt(qx**2+qy**2);
        if qx == 0.0:
            angl = np.pi/2;
        else:
            angl = np.arctan(qy/qx);
        return self.getIntermediateTunes(ampl, angl);
    
    #
    #WARNING use with extreme care, this meant for a very specific type of footprint
    #
    def repair(self,orders=[1,2,3,4,5],tol=1E-4):
        x = [];
        y = [];
        for ampl in range(self.getNbAmpl()):
            for angl in range(self.getNbAngl()):
                qx = self.getHTune(ampl,angl);
                qy = self.getVTune(ampl,angl);
                newAngl = angl;
                backwards = True;
                while self._onAnyRes(qx, qy, orders,tol=tol):
                    if backwards:
                        if newAngl == 0:
                            backwards = False;
                            newAngle = angl;
                        else:
                            newAngl = newAngl - 1;
                    if not backwards:
                        newAngl = newAngl + 1;
                        if newAngl >= self.getNbAngl():
                            print("ERROR could not find a non resonant tune at this amplitude, leaving original value");
                            newAngle = angl;
                            break;
                    qx = self.getHTune(ampl,newAngl);
                    qy = self.getVTune(ampl,newAngl);
                if newAngl != angl:
                    self._tunes[ampl][angl]['H'] = qx;
                    self._tunes[ampl][angl]['V'] = qy;
                    x.append(qx);
                    y.append(qy);
        return [x,y];
                
    def _onAnyRes(self,qx,qy,orders,tol=1E-4):
        for order in orders:
            if self._onRes(qx,qy,order,tol=tol):
                return True;
        return False;
    
    def _onRes(self,qx,qy,order,tol=1E-4):
        for n in range(1,order+1):
            m = order-n;
            value = (n*qx+m*qy);
            if np.abs(value-np.floor(value+0.5))<tol:
                return True;
            value = (n*qx-m*qy);
            if np.abs(value-np.floor(value+0.5))<tol:
                return True;
        return False;
