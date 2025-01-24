from .trainer import BaseTrainer, CRDTrainer, AugTrainer, DynamicTemperatureScheduler

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "ls": AugTrainer,
    "scheduler": DynamicTemperatureScheduler,
}
