from classispecies import settings

# settings.SOUNDPATHS.update({
#     'Boa' :
#         { 'train' : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train',
#           #'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train',
#           'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/test',
#         },
#     })
# settings.LABELS = {
#     'Boa' :
#         { 'train' : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/numero_file_train.csv',
#           #'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/numero_file_train.csv',
#           'test'  : '',
#         },
#     }


settings.modelname  = "nips4b"
settings.classifier = "decisiontree" #randomforest"
settings.analyser   = "hertzfft" #"melfft" #hertz-spectrum" #"oskmeans" 

settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = True
settings.FORCE_FEATXTRALL = True
settings.MULTILABEL = True
settings.MULTICORE  = False

settings.extract_mean = False
settings.extract_std  = False

settings.sec_segments = 0.5
settings.n_segments   = None

settings.FEATURES_PLOT = False


from classispecies.classispecies import Classispecies, multirunner
from classispecies.utils import misc

settings.LABELS = {'default' : {
   'train' : settings.modelname + "-train.csv",
   'test'  : settings.modelname + "-test.csv",
}}

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
        
#model = NipsModel(analyser=settings.analyser, classifier=settings.classifier)
#model.run()
#model.dump_to_nips4b_submission()
#model.save_to_db()

# settings.analyser = "multiple"
# settings.extract_mel   = False
# settings.extract_dolog = False
# settings.extract_dct   = False
# settings.extract_fft2  = True
# settings.extract_max  = False
#    
#    
#     
# settings.sec_segments = 0.5
# settings.n_segments   = None
# settings.MULTICORE = False
# settings.FORCE_FEATXTRALL = True
# settings.FORCE_FEATXTR = True
#     
# settings.FEATURE_ONEFILE_PLOT = True
#     
# for ds_ in [10, 32, 64]:
#     settings.downscale_factor = ds_   
#     model = NipsModel(analyser=settings.analyser, classifier=settings.classifier)
#     model.run()

settings.downscale_factor = 50
settings.FORCE_MULTIRUNNER_RECOMPUTE = False
settings.SAVE_TO_DB = True

model = multirunner(NipsModel, [None], iters=1)
#model = multirunner(NipsModel, [1.0], iters=1)
