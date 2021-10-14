# dependencies for file reading
import json
import sys
import itertools
import numpy as np
import os
import soundfile as sf
import math

import librosa # conda install -c conda-forge librosa

# make script paths from one level up avaialble for import
script_path = os.path.realpath(os.path.dirname(__name__))
os.chdir(script_path)
sys.path.append(script_path)

from parammanager import paramManager
from nsjsonmanager import nsjson


from genericsynth import synthInterface as SI
from filewrite import fileHandler

import importlib


'''
This code will generate a dataset of textures consiting of pop or drip textures. 

The files are generated using 3 (or N) different parameters that are sampled over a range of values. The parameters that are 
developed for the sound model are exposed via the config_file.json. The three parameters for a range of sounds
including the pop and drip textures are:
    rate (average events per second),
    irregularity in temporal distribution (using a gaussian distribution around each evenly-spaced time value), and
    the center frequency of bp filter

The generator.py to be independent of any synth, and dependent only on the config file. That is, the same generator.py should work for all DSSynths.
    a) It can get the synth name from the config file, and then import it "dynamically"
    b) It can set any params the user wants to fix (not iterate over)
    c) names in the config file should correspond to names in the synth (right now the generator constructs synth param names from those use in the cofig file by adding "_exp", etc.

There is also a "visualizer" notebook need not generate files at all. The function of the visualizer is to 
interactively explore and create textures using synthinterface and sound models.
It is mostly for understanding the synthesizer, and exploring parameters that you might help you decide how 
you want to specify them in your config file.

The parameter values are each sampled liniearly on an exponential scale, and specified in:
rate = 2^r_exp  (so r_exp in [0,4] means the rate ranges from 2 to 16)
irregularity = .04*10^irreg_exp; sd = irregularity/events per second  (so irreg_exp in [0,1] means irregularity ranges from completely regular, to Poisson process)
cf = 440*2^cf_exp  (so cf_exp in [0,1] means cf ranges from 440 to 880, one octave)

Generator use:
For each parameter setting, first a "long" signal (of lentgth longDurationSecs) is generated, and then
it is sliced into segments (called variations) of a length desired for training.

Example: If each parameter is sampled at 5 values, the long signal is 10 seconds and variationLength is 2 seconds,
then The the total amount of audio generated is 5*5*5*10= 1250 seconds of sound (about 25 hours; ~3Gb at 16K sr).
If each variation is 2 seconds, then there will be 10/2=5 variations for each parameter setting, and
5*5*5*5 = 625 files
'''

import argparse

myConfig = {}
soundModels = {}
outputpath = ""

def get_arguments():
    parser = argparse.ArgumentParser(description="myParser")
    parser.add_argument("--configfile", required=True)
    parser.add_argument("--outputpath", required=True)
    return parser.parse_args()

def main():

    # folderConsistency()

    args = get_arguments()
    module_name = args.configfile # here, the result is the file name, e.g. config or config-special
    outputpath = args.outputpath

    # Not use __import__, use import_module instead according to @bruno desthuilliers's suggestion
    # __import__(module_name) # here, dynamic load the config module
    # MyConfig = sys.modules[module_name].MyConfig # here, get the MyConfig class
    # MyConfig = importlib.import_module(module_name)

    with open(module_name) as json_file:
        MyConfig = json.load(json_file)
        print("Reading parameters for generating ", "red" , MyConfig['soundname'], " texture.. ")
        # for p in MyConfig['params']:
        #     p['formula'] = eval("lambda *args: " + p['formula'])
        # for p in MyConfig['fixedParams']:
        #     p['formula'] = eval("lambda *args: " + p['formula'])        

    loadSoundModels(MyConfig)
    MyConfig["outputpath"] = outputpath

    # from args.configfile import MyConfig # <-- how is that possible?
    generate(MyConfig)

    # print(MyConfig["params"])

def loadSoundModels(MyConfig):
    dirpath = os.getcwd()
    # modules = [f for f in os.listdir(os.path.dirname(dirpath)) if f[0] != "." and f[0] != "_"]
    # for module in modules:
    spec = importlib.util.spec_from_file_location(dirpath, os.path.join(dirpath,MyConfig["soundname"]+".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    soundModels["sound"] = mod
    # mod_name = file[:-3]   # strip .py at the end
    # exec('from soundModels' + ' import ' + os.path.abspath(mod_name))

    # files = [f for f in os.listdir(os.path.dirname(dirpath+module+"/")) if f[0] != "." and f[0] != "_"]
    #for file in files:
    #    spec = importlib.util.spec_from_file_location(module, os.path.join(dirpath,module +"/" + file))
    #    mod = importlib.util.module_from_spec(spec)
    #    spec.loader.exec_module(mod)
    # importlib.import_module(dirpath + directory)

def generate(MyConfig):
    
    '''Initializes file through a filemanager'''
    fileHandle = fileHandler()
    # MyConfig["outPath"]
    dirpath = "/"
    outputpath = MyConfig["outputpath"]

    if os.path.isdir(outputpath):
        print("Outpath exists")
    else:
        print(outputpath)
        os.mkdir(outputpath)

    print("Enumerating parameter combinations..")

    '''
        for every combination of cartesian parameter
        for every variation
            Create variation wav files
            Create variation parameter files
    '''

    '''2 arrays for normalised and naturalised ranges'''
    userRange = []
    synthRange = []
    paramArr = MyConfig["params"]
    fixedParams = MyConfig["fixedParams"]

    for p in MyConfig["params"]:
            userRange.append(np.linspace(p["user_minval"], p["user_maxval"], p["user_nvals"], endpoint=True))
            synthRange.append(np.linspace(p["synth_minval"], p["synth_maxval"], p["user_nvals"], endpoint=True))
        
    userParam = list(itertools.product(*userRange))
    synthParam = list(itertools.product(*synthRange))

    numChunks=MyConfig["numChunks"]
    #math.floor(MyConfig["soundDuration"]/MyConfig["chunkSecs"])  #Total duraton DIV duraiton of each chunk 

    totalDuration = len(userParam)*MyConfig["soundDuration"] # Total duration of the audio textures generated for this dataset'''
 
    '''Set fixed parameters prior to the generation'''
    # print(soundModels[MyConfig["soundname"]].PatternSynth)
    barsynthclass = getattr(soundModels["sound"],MyConfig["soundname"])

    '''use a sample rate and rng seed if initialized'''
    if not "rngseed" in MyConfig:
        barsynth= barsynthclass(sr=MyConfig["computeSR"])
    elif MyConfig["rngseed"] == None:
        ''' Generate random seed '''
        seed = np.random.randint(0, np.power(2,32))
        print("Using user random seed", seed)
        barsynth= barsynthclass(sr=MyConfig["computeSR"], rngseed=seed)
    else:
        print("Using user random seed", MyConfig["rngseed"])
        barsynth= barsynthclass(sr=MyConfig["computeSR"], rngseed=MyConfig["rngseed"])
    print(barsynth)

    # Manually set the parameters to Natural    
    for params in paramArr:
        params["synth_units"] = "natural"

    for fixparams in fixedParams:
        '''Setting in natural ranges'''
        fixparams["synth_units"] = "natural"
        barsynth.setParam(fixparams["synth_pname"], fixparams["synth_val"])
    
    sg = nsjson.nsJson("/", outputpath, 1, MyConfig["datafileSR"], MyConfig['soundname'])

    '''Only initialize if record is in tfrecord format'''
    if MyConfig["recordFormat"] == "tfrecords":
        try:
            from tfrecordmanager import tfrecordManager
            tfr=tfrecordManager.tfrecordManager()
        except:
            print("Please install tfrecords with <pip install -r requirements_tf.txt --src '.'> and run again")
            sys.exit()

    '''files for agrregate tfrecord writing'''
    audioSegments = []
    pfnames = []
    soundDurations = []
    segmentNum = []
    usertfP = []
    synthtfP = []

    '''Enumerate parameters'''
    for index in range(len(userParam)): # iterating through a caretesian product of lists

        '''Stepping through enumerated dataset'''
        userP = userParam[index]
        synthP = synthParam[index]

        for paramInd in range(len(paramArr)):
            # if paramArr[paramInd]["synth_units"] == "norm":
            #     '''Setting in Normal ranges'''
            #     barsynth.setParamNorm(paramArr[paramInd]["synth_pname"], synthP[paramInd])
            # else: 
            '''Setting in natural ranges'''
            barsynth.setParam(paramArr[paramInd]["synth_pname"], synthP[paramInd])
        
        barsig=barsynth.generate(MyConfig["soundDuration"])
        numChunks=MyConfig["numChunks"]#math.floor(MyConfig["soundDuration"]/MyConfig["chunkSecs"])  #Total duraton DIV duraiton of each chunk 
        chunkSecs = MyConfig["soundDuration"]/numChunks

        for v in range(numChunks):

            fileHandle = fileHandler()

            if numChunks == 1:
                '''Write wav'''
                #wavName = fileHandle.makeName(MyConfig["soundname"], paramArr, fixedParams, userP, v)
                wavName = fileHandle.makeName(MyConfig["soundname"], paramArr, userP, None)
                wavPath = fileHandle.makeFullPath(outputpath,wavName,".wav")
                chunkedAudio = SI.selectVariation(barsig, MyConfig["computeSR"], v, chunkSecs)

                if MyConfig["computeSR"] != MyConfig["datafileSR"]:
                    newsig=librosa.resample(chunkedAudio, MyConfig["computeSR"], MyConfig["datafileSR"])
                    sf.write(wavPath, newsig, MyConfig["datafileSR"])
                else:
                    newsig = chunkedAudio
                    sf.write(wavPath, chunkedAudio, MyConfig["datafileSR"])

                '''Write params'''
                #paramName = fileHandle.makeName(MyConfig["soundname"], paramArr, fixedParams, userP, v)
                paramName = fileHandle.makeName(MyConfig["soundname"], paramArr, userP, None)
                pfName = fileHandle.makeFullPath(outputpath, paramName,".params")
            else:
                wavName = fileHandle.makeName(MyConfig["soundname"], paramArr, userP, v)
                wavPath = fileHandle.makeFullPath(outputpath,wavName,".wav")
                chunkedAudio = SI.selectVariation(barsig, MyConfig["computeSR"], v, chunkSecs)

                if MyConfig["computeSR"] != MyConfig["datafileSR"]:
                    newsig=librosa.resample(chunkedAudio, MyConfig["computeSR"], MyConfig["datafileSR"])
                    sf.write(wavPath, newsig, MyConfig["datafileSR"], subtype='PCM_16')
                else:
                    newsig = chunkedAudio
                    sf.write(wavPath, chunkedAudio, MyConfig["datafileSR"], subtype='PCM_16')

                '''Write params'''
                #paramName = fileHandle.makeName(MyConfig["soundname"], paramArr, fixedParams, userP, v)
                paramName = fileHandle.makeName(MyConfig["soundname"], paramArr, userP, v)
                pfName = fileHandle.makeFullPath(outputpath, paramName,".params")

            if MyConfig["recordFormat"] == "params" or MyConfig["recordFormat"]==0:
                pm=paramManager.paramManager(pfName, fileHandle.getFullPath())
                pm.initParamFiles(overwrite=True)

                '''Write parameters and meta-parameters'''
                for pnum in range(len(paramArr)):
                        #pm.addParam(pfName, paramArr[pnum]['synth_pname'], [0,MyConfig["soundDuration"]], [userP[pnum], userP[pnum]], units=paramArr[pnum]['synth_units'], nvals=paramArr[pnum]['user_nvals'], minval=paramArr[pnum]['user_minval'], maxval=paramArr[pnum]['user_maxval'], origUnits=None, origMinval=paramArr[pnum]['synth_minval'], origMaxval=paramArr[pnum]['synth_maxval'])
                        pm.addParam(pfName, paramArr[pnum]['synth_pname'], [0,chunkSecs], [userP[pnum], userP[pnum]], units=paramArr[pnum]['synth_units'], nvals=paramArr[pnum]['user_nvals'], minval=paramArr[pnum]['user_minval'], maxval=paramArr[pnum]['user_maxval'], origUnits=None, origMinval=paramArr[pnum]['synth_minval'], origMaxval=paramArr[pnum]['synth_maxval'])
                        
                        if "user_doc" in paramArr[pnum] and paramArr[pnum]["user_doc"] != "" :
                            pm.addMetaParam(pfName, paramArr[pnum]['synth_pname']+"_user_doc",paramArr[pnum]['user_doc']) 

                        pm.addMetaParam(pfName, paramArr[pnum]['synth_pname']+"_synth_doc",barsynth.getParam(paramArr[pnum]["synth_pname"],"synth_doc"))
                
                for pnum in range(len(fixedParams)):
                    #pm.addParam(pfName, fixedParams[pnum]['synth_pname'], [0,MyConfig["soundDuration"]], [fixedParams[pnum]["synth_val"], fixedParams[pnum]["synth_val"]], units=fixedParams[pnum]['synth_units'], nvals=2, origUnits=None)
                    if "user_doc" in fixedParams[pnum] and fixedParams[pnum]["user_doc"] != "" :
                        pm.addMetaParam(pfName, fixedParams[pnum]['synth_pname']+"_user_doc",fixedParams[pnum]['user_doc']) 

                    pm.addMetaParam(pfName, fixedParams[pnum]['synth_pname']+"_synth_doc",barsynth.getParam(fixedParams[pnum]["synth_pname"],"synth_doc"))

            elif MyConfig["recordFormat"] == "nsjson" or MyConfig["recordFormat"] == 1:
                
                sg.storeSingleRecord(wavName)
                for pnum in range(len(paramArr)):
                    sg.addParams(wavName, paramArr[pnum]['synth_pname'], userP[pnum], barsynth.getParam(paramArr[pnum]['synth_pname']))
                sg.write2File("nsjson.json")
            
            elif MyConfig["recordFormat"] == "tfrecords":

                tfr.__addFeatureData__(pfName, [0, chunkSecs], newsig, v)
                # MyConfig["shard_size"], MyConfig["samplerate"], totalDuration)
                
                for pnum in range(len(paramArr)):
                    # paramArr[pnum]['synth_units'], paramArr[pnum]['user_nvals'], paramArr[pnum]['user_minval'], paramArr[pnum]['user_maxval'], paramArr[pnum]['synth_minval'], paramArr[pnum]['synth_maxval']
                    tfr.__addParam__(paramArr[pnum], userP[pnum])

                for pnum in range(len(fixedParams)):
                    tfr.__addParam__(fixedParams[pnum], fixedParams[pnum]["synth_val"])

                #print("size is " , tfr.__tfRetSize__())
                tfr.__tfUpdateSize__()

                '''Usage of tfrecords with single record per file'''                
                if MyConfig["tftype"] == "single":                                
                    tfr.__tfwriteOne__(pfName)
                    print("Generated a tfrecord")
                else:

                    ''' Append and do not write'''
                    if tfr.__tfRetSize__() < MyConfig["shard_size"] :
                        #print("new size with record " , tfr.__tfRetSize__())
                        audioSegments.append(newsig)
                        pfnames.append(pfName)
                        soundDurations.append([0,chunkSecs])
                        segmentNum.append(v)
                        usertfP.append(userP)
                        synthtfP.append(synthtfP)
                    else:
                        ''' Write and then append'''
                        print(len(pfnames))
                        tfr.__tfwriteN__(outputpath, pfnames, soundDurations, segmentNum, audioSegments, usertfP, synthtfP, paramArr, fixedParams)
                        audioSegments = []
                        pfnames = []
                        soundDurations = []
                        segmentNum= []
                        usertfP = []
                        synthtfP = []
                        # afresh with current record
                        audioSegments.append(newsig)
                        pfnames.append(pfName)
                        soundDurations.append([0,chunkSecs])
                        segmentNum.append(v)
                        usertfP.append(userP)
                        synthtfP.append(synthtfP)

                        tfr.__addFeatureData__(pfName, [0, chunkSecs], newsig, v)
                        # MyConfig["shard_size"], MyConfig["samplerate"], totalDuration)
                        
                        for pnum in range(len(paramArr)):
                            # paramArr[pnum]['synth_units'], paramArr[pnum]['user_nvals'], paramArr[pnum]['user_minval'], paramArr[pnum]['user_maxval'], paramArr[pnum]['synth_minval'], paramArr[pnum]['synth_maxval']
                            tfr.__addParam__(paramArr[pnum], userP[pnum])

                        for pnum in range(len(fixedParams)):
                            tfr.__addParam__(fixedParams[pnum], fixedParams[pnum]["synth_val"])

                        #print("size is " , tfr.__tfRetSize__())
                        tfr.__tfUpdateSize__() #might be a problem in edge case when each record is as big as max tfrecord size.
                        #print("Updated size is " , tfr.__tfRetSize__())
                    
                    if index == (len(userParam) - 1) and v == (numChunks-1):
                        print(len(pfnames))
                        tfr.__tfwriteN__(outputpath, pfnames, soundDurations, segmentNum, audioSegments, usertfP, synthtfP, paramArr, fixedParams)

            else:
                print("Not recognized format")

    # if MyConfig["recordFormat"] == "tfrecords" and MyConfig["tftype"] == "shards":

    #     tfr=tfrecordManager.tfrecordManager()

    #     print("Number of records per shard", len(pfnames))

    #     pfName = fileHandle.makeFullPath(outputpath, "shard"+str(fileid),".params")
    #     outrecord = pfName.split(".params")[0]+'.tfrecord'

    #     tfr.__tfwriteN__(outrecord, pfnames, soundDurations, segmentNum, audioSegments, userParam, synthParam, paramArr, fixedParams, numChunks, beg, end)

    #tfm=tfrecordManager.tfrecordManager(vFilesParam[v], outPath)
    #data,sr = librosa.core.load(outPath + fname + '--v-'+'{:03}'.format(v)+'.wav',sr=16000)
    #print(len(data))
    #tfm.addFeature(vFilesParam[v], 'audio', [0,len(data)], data, units='samples', nvals=len(data), minval=0, maxval=0)
    #for pnum in range(len(paramArr)):
    #   print(pnum)
    #   tfm.addFeature(vFilesParam[v], paramArr[pnum]['pname'], [0,data['soundDuration']], [enumP[pnum], enumP[pnum]], units=paramArr[pnum]['units'], nvals=paramArr[pnum]['nvals'], minval=paramArr[pnum]['minval'], maxval=paramArr[pnum]['maxval'])
    #tfm.writeRecordstoFile()

if __name__ == '__main__':
    main()