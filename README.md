The python file imgClassifier.py was used to complete answers.pdf. The help information can be accessed using the flag --help and is displayed below.

imgClassifier.py by T. J. Tkacik
        
        Accepted flags:

        --help    for this help information
        -l        for loud output, default False
        -f        to select folder, default 'images'

        Example:   imgClassifier.py -l -f images
         
        Note:   imgClassifier.py is prone to fail by resulting in a
                segmentation fault (core dump). Because of this, models
                that are successfully generated are saved as class_kernal.model
                files and used in later runs. To retrain the parameters of an 
                SVM, delete the model files from this directory.