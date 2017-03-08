## Objective


Broadcasters and cultural heritage organizations have digitized thousands of hours of archival audio in recent years, in an attempt to save recordings from fragile analog storage media. Depending on budget limitations constraints and intended uses, available metadata varies widely in its scope and quality. As digital audio collections continue to grow, institutions are just beginning to grapple with the challenge of making them accessible and useful.

Annotating audio by hand is time-consuming. Tasks such as labeling speakers, indicating structural features within recordings, and generating topic summaries are slow enough that they don't scale well to large collections. On the other hand, attempts to automate such processes using machine learning produce data that can't be taken as entirely reliable.
One approach, then, is to use imperfect machine classifications to speed up the process of human annotation. As we add audio to our human-generated and human-checked dataset, we can then update our classifiers and iteratively improve their performance.

To date, relatively little work has been done on  applying speaker recognition tools in real-world digital archives. Freely available software tools for speaker recognition tend to be created by researchers in machine learning and linguistics; most require a high degree of technical sophistication to configure and get running. Documentation is often limited, and many usage demonstrations assume users have access to proprietary speech audio databases. In short, there is no off-the-shelf tool that will handle all the steps in our speaker recognition pipeline.

Existing tools require users to provide their own systems for generating and managing audio data in the file system. To speed up such tasks we have introduced Audio Tagging Toolkit, a collection of Python scripts for completing batch audio processing tasks.




## Relevant Projects

- INA
  - highly developed labeling project, but all closed source
- BBC work
  - topic extraction based on speech-to-text conversion
  - speaker segmentation with human labeling
  - tests with audio-based speaker recognition, but not yet applied on a large scale
  - using fre and open source tools
  - work divided between independent applications which can be linked together in sequence (Raimond et al. 2012)

- AVPreserve
  - discusses metadata generation mechanisms (MGMs) as applied in 3 stages, each building upon outpyt from the previous stage (Lacinak and Dunn 2016)


- Lessons: 
  - These systems can be applied in layers; one metatadata generation step feeds into the next, as well as back into itself.
  - State of the art changes fast; different features needed for different types of classification and metadata generation.




## ML Landscape

Speaker recognition (sometimes called speaker identification) is an active topic in machine learning research, comprising several classes of problems: Text-dependent classifiers use human- or machine-generated transcripts to help make classifications; text-independent systems rely solely on audio data. Closed-set classifiers expect that each passage in a given recording can be classified as one of several individuals used to train the model, while open-set systems attempt to locate a single individual as compared to a training set made up of audio from many different speakers.

Open-set, text-independent speaker recognition is the most difficult problem in the subfield, made more difficult in the context of noisy, unpredictable data in real-world collections. When training classifiers for this task, ML researchers have used two approaches: universal background models (UBM) and cohort models. The former compares audio of a given speaker to a broad, perhaps randomly selected, collection of other speech recordings. Cohort models, by contrast, are trained using voices known to sound similar to the speaker in question in an attempt to improve discriminative subtlety.

Most state-of-the-art speaker recognition systems follow one of two paradigms: i-vectors and deep neural nets. Under the i-vector approach, which gained popularity in the early 2010s (Dehak et al. 2011), low-dimensional vectors (i-vectors or "identity vectors") extracted from low-level audio features using factor analysis, are used to train classifiers, with support vector machines (SVM) and gaussian mixture models (GMM) as the most popular options. Recent work suggests deep neural nets (DNNs) can outperform such i-vector systems; alternately, DNNs may be trained using on extracted i-vector data.



## Software landscape

There are various tools available for each processing step in a speaker recognition pipeline, any of which may be used interchangeably or in combination. I will now describe some salient examples.

- Sonic Visualiser is a GUI-based tool for viewing and annotating audio recordings. As an actively developed software project available on all major operating systems, we have chosen this tool as our go-to for editing and viewing annotation data.

  - [image: human-tagged CSV overlaid on audio file]

- Machine learning toolkits that have been used for speaker recognition tasks include Kaldi, SIDEKIT, ALIZE, and bob.bio.spear. Each has its limitations, and virtually all documentation assumes that a user has access to proprietary datasets such as NIST's "Speaker Recognition Evaluation" series.

- Of these options, SIDEKIT appears most likely to be useful for our purposes. It is designed to handle each step in the pipeline, including preprocessing, feature extraction, and hyperparameter tuning. In addition to training SVMs and GMMs, SIDEKIT will soon add support for the DNN tool Theano (Larcher 2016).

- pyAudioAnalysis is a high-level toolkit for audio machine learning in Python, including pre-trained models for preprocessing steps such as speaker segmentation and music/speech classification. Given two batches of audio in WAV format, pyAudioAnalysis extracts a wide range of audio features and handles dimension reduction and hyperparameter tuning as it trains one of five models: SVM, k-nearest neighbor (KNN), random forests, gradient boosting, and extra trees. Other than KNN, all models are trained using scikit-learn.

Though pyAudioAnalysis is not designed for speaker recognition in particular, its ease of use has made it a good first step in the bootstrapping process. Once we have a sufficient quantity of training data, we can swap in other tools to train more refined classifier models.



## Audio Tagging Toolkit

Because each ML tool plays one part in a longer pipeline, system designers are expected to generate and manage their own annotation and audio datasets. In order to expedite these common processes, we have developed a collection of batch processing scripts titled Audio Tagging Toolkit.

- Diarize.py
  - Applies speaker segmentation using model included in pyAudioAnalysis.
  - Guesses total number of speakers and assigns a speaker class to each audio segment.
  - Includes single-file and batch modes.
  - A human can add speaker labels and rapidly edit/remove identified segments in Sonic Visualiser.

- FindApplause.py
  - Identifies applause in recordings using an SVM model trained on data from the PennSound poetry archive and the AAPB.
  - Includes single-file and batch modes.
  - When applause appears between structural units in a recording, classifier output can help speed up speaker labeling.

- QuickCheck.py
  - Workflow tool for rapidly checking and correcting classifier output.

- RandomTags.py
  - Applies annotation tags at random to one or more audio files.
  - Randomly selected segments can be used to train background models.

- ExcerptClass.py
  - Batch extracts specified WAV segments from a given audio file.
  - Segments to be extracted are specified in a CSV.

- EvaluateAudio.py
  - Simple GUI tool for playing snippets of classified audio and evaluating whether they are labeled correctly.


## Workflow 

- Human tagging

- ML pipeline
  - Preprocessing
    - Mid-pass filter, downsampling, etc.
    - Remove music and silence
    - Low-level feature extraction
    - Dimension reduction

  - Classification
    - Training model(s), inclyding hyperparameter tuning
    - Running model(s) on new audio

  - Postprocessing
    - Smoothing
    - Combining weighted output from several models

- More human tagging, then iterate.

  ​



## Work coming next


- Add music/speech classification as another preprocessing step.

- Design standard format for documenting and publishing trained models and extracted features.

- Work to implement up-to-date audio ML software (most likely SIDEKIT).