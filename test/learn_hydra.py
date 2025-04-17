import hydra  
from omegaconf import DictConfig, OmegaConf
import os

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(version_base=None, config_path=FILE_PATH, config_name="sim") 
def my_app(cfg: DictConfig)->None:
    print(OmegaConf.to_yaml(cfg))
    model_name:str = cfg.train_setting.model
    print("model_name: ",model_name)

if __name__ == "__main__":
    my_app()

