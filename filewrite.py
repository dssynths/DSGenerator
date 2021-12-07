

# loosely inspired from the parammanager
# Takes a file name and path and returns a full path to file.
import os

class fileHandler():

    def __init__(self) :

        self.fname=""
        self.outpath = ""

    '''make file name from soundname, paramNames, param values, chunk number, example number'''
    def makeName(self, soundName, paramArr, enumP, chunk, examples=1, x=1):
        '''Construct filenames with static parameters'''
        self.fname = soundName
        # for paramNum in range(len(fixedParams)):
        #     self.fname = self.fname + '--' + fixedParams[paramNum]['synth_pname'] + '-'+'{:05.2f}'.format(fixedParams[paramNum]["synth_val"])
        for paramNum in range(len(paramArr)):
            self.fname = self.fname + '--' + paramArr[paramNum]['user_pname'] + '-'+'{:05.2f}'.format(enumP[paramNum])
        if chunk != None:
            self.fname = self.fname + '--c-'+'{:02}'.format(chunk)
        if examples > 1:
            self.fname = self.fname + '--x-'+'{:02}'.format(x)

        return self.fname

    def getFileName(self):
        return self.fname

    def getFullPath(self):
        return self.outpath

    # only store the actual value, not the normed value used for setting
    def makeFullPath(self, outpath, name, ext) :

        if os.path.isdir(outpath):
            self.outpath = outpath
        else:
            self.outpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), outpath)
            if not os.path.isdir(self.outpath):
                os.mkdir(self.outpath)

        fullpath = os.path.join(outpath, name + ext)
        return fullpath
