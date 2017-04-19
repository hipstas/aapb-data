import os
from pyAudioAnalysis import audioTrainTest as aT

os.chdir('/export/home/u16/smcl/AAPB')


aT.featureAndTrain(['Reagan','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm_Reagan_UBM_all", False)

aT.featureAndTrain(['Reagan','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gradientboosting_Reagan_UBM_all", False)

aT.featureAndTrain(['Reagan','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "extratrees_Reagan_UBM_all", False)

aT.featureAndTrain(['Reagan','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "randomforest_Reagan_UBM_all", False)

aT.featureAndTrain(['Reagan','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn_Reagan_UBM_all", False)




import os
from pyAudioAnalysis import audioTrainTest as aT

os.chdir('/export/home/u16/smcl/AAPB')


aT.featureAndTrain(['Child','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm_Child_UBM_all", False)

aT.featureAndTrain(['Child','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gradientboosting_Child_UBM_all", False)

aT.featureAndTrain(['Child','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "extratrees_Child_UBM_all", False)

aT.featureAndTrain(['Child','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "randomforest_Child_UBM_all", False)

aT.featureAndTrain(['Child','test_set_616_clips'], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn_Child_UBM_all", False)



















from pyAudioAnalysis import audioSegmentation as aS
[flagsIndknn, classesAll, acc, CM] = aS.mtFileClassification("/Users/mclaugh/Desktop/MLK_2/Martin Luther King   The Three Evils of Society-j8d-IYSM-08.WAV", "/Volumes/McLaughlin-6TB-1/Dropbox/test_set_616_clips/knn_MLK_bg", "knn", True)











############



from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "etMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gbMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "rfMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm5Classes")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn5Classes")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "et5Classes")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gb5Classes")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "rf5Classes")