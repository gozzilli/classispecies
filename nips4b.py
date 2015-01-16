from classispecies import settings

settings.SOUNDPATHS.update({
    'Boa' :
        { 'train' : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train',
          #'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train',
          'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/test',
        },
    })
settings.LABELS = {
    'Boa' :
        { 'train' : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/numero_file_train.csv',
          #'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/numero_file_train.csv',
          'test'  : '',
        },
    }


settings.modelname  = "nips4b"
settings.classifier = "randomforest"
settings.analyser   = "hertzfft" #"melfft" #hertz-spectrum" #"oskmeans" 

settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = True
settings.MULTILABEL = True
settings.MULTICORE  = True

settings.extract_mean = True
settings.extract_std  = False

settings.sec_segments = 1.0
settings.n_segments   = None


from classispecies.classispecies import Classispecies
from classispecies.utils import misc

class NipsModel(Classispecies):
    
    def load_labels(self, min_row=0, max_row=None, min_col=0, max_col=None):
        return super(NipsModel, self).load_labels(0, None, 1, -1)
        

    def dump_to_nips4b_submission(self):
        with open(misc.make_output_filename("nips4b_bird2013_sidelil_3", "submission", settings.modelname, "csv"), 'w') as f:
            f.write("ID,Probability\n")
            nrow=1
            for row in self.predict_proba:
                ncol=1
                for column in row:
                    f.write("nips4b_birds_testfile%04d.wav_classnumber_%d,%f\n" %(nrow, ncol, column))
                    ncol+=1
                nrow+=1

        with open(misc.make_output_filename("nips4b_bird2013_sidelil_4", "submission", settings.modelname, "csv"), 'w') as f:
            f.write("ID,Probability\n")
            nrow=1
            for row in self.res_:
                ncol=1
                for column in row:
                    f.write("nips4b_birds_testfile%04d.wav_classnumber_%d,%f\n" %(nrow, ncol, column))
                    ncol+=1
                nrow+=1
                
#    def dump_to_nips4b_submission(self):
#        with open(misc.make_output_filename("nips4b_bird2013_sidelil_2", "submission", settings.modelname, "csv"), 'w') as f:
#            f.write("ID,Probability\n")
#            nrow=1
#            for row in self.prediction:
#                ncol=1
#                for column in row:
#                    f.write("nips4b_birds_testfile%04d.wav_classnumber_%d,%f\n" %(nrow, ncol, column))
#                    ncol+=1
#                nrow+=1
        


model = NipsModel(analyser=settings.analyser, classifier=settings.classifier)
model.run()
model.dump_to_nips4b_submission()