import os
import zipfile
import pickle


class Loader:
    """
    Util class because the saving in stable baselines is not very flexible.
    It dumps a zip file, so if we want to have a single log file we need to further zip it along a pickle.
    """

    def __init__(self, model=None, args=None):
        self.model = model
        self.args = args
        self.possible_models = {'a2c', 'sac'}

    def save(self, name):
        modelname = f"{name}_model"
        paramname = f"{name}_param.p"
        zipname = f"{name}.zip"

        self.model.save(modelname)
        pickle.dump(self.args, open(paramname, 'wb'))
        z = zipfile.ZipFile(zipname, 'w')
        with z:
            z.write(f'{modelname}.zip')
            z.write(paramname)
        os.remove(paramname)
        os.remove(f'{modelname}.zip')

    def load(self, name):
        modelname = f"{name}_model"
        paramname = f"{name}_param.p"
        zipname = f"{name}.zip"

        with zipfile.ZipFile(zipname, 'r') as z:
            z.extractall('.')
        args = pickle.load(open(paramname, 'rb'))
        model_type = args['model'].lower()
        assert model_type in self.possible_models
        if model_type == 'a2c':
            from stable_baselines3 import A2C
            model = A2C.load(modelname)
        elif model_type == 'sac':
            from stable_baselines3 import SAC
            model = SAC.load(modelname)
        else:
            model = 0

        os.remove(paramname)
        os.remove(f'{modelname}.zip')

        self.model = model
        self.args = args
        return model, args
