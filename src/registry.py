#registry as per AISI

THREAT_MODELS = {}
UNLEARNERS = {}
SCORERS = {}


#register threat model class
def register_threat_model(name):
    def deco(cls):
        THREAT_MODELS[name] = cls
        return cls
    return deco


#register unlearner class
def register_unlearner(name):
    def deco(cls):
        UNLEARNERS[name] = cls
        return cls
    return deco


#register scorer builder func
def register_scorer(name):
    def deco(fn):
        SCORERS[name] = fn
        return fn
    return deco


#build threat model from config selection
def build_threat_model(config):
    cls = THREAT_MODELS[config.threat_model]
    return cls(
        target_label=config.target_label,
        poison_ratio=config.poison_ratio,
        **config.threat_model_args,
    )


#build unlearner from config selection
def build_unlearner(config):
    cls = UNLEARNERS[config.unlearner]
    return cls(**config.unlearner_args)


#build scorer list from config selection
def build_scorers(config):
    return [SCORERS[name](config) for name in config.scorers]


#import implementation modules
def import_builtins():
    from src.threat_models import fudge
    from src.unlearning import pga
    from src.audit import scorers
