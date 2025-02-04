from .trainer import BaseTrainer, CRDTrainer, AugTrainer, DynamicTemperatureScheduler, DynamicAugTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ls": AugTrainer,
    "scheduler": DynamicTemperatureScheduler,
    "ourls": DynamicAugTrainer,
}
